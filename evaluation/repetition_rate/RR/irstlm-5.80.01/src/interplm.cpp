/******************************************************************************
IrstLM: IRST Language Model Toolkit
Copyright (C) 2006 Marcello Federico, ITC-irst Trento, Italy

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

******************************************************************************/

using namespace std;

#include <cmath>
#include "mfstream.h"
#include "mempool.h"
#include "htable.h"
#include "dictionary.h"
#include "n_gram.h"
#include "mempool.h"
#include "ngramcache.h"
#include "ngramtable.h"
#include "normcache.h"
#include "interplm.h"


void interplm::trainunigr()
{

  int oov=dict->getcode(dict->OOV());
  cerr << "oovcode: " << oov << "\n";

  if (oov>=0 && dict->freq(oov)>= dict->size()) {
    cerr << "Using current estimate of OOV frequency " << dict->freq(oov)<< "\n";
  } else {
    oov=dict->encode(dict->OOV());
    dict->oovcode(oov);

    //choose unigram smoothing method according to
    //sample size
    //if (dict->totfreq()>100){ //witten bell
    //cerr << "select unigram smoothing: " << dict->totfreq() << "\n";

    if (unismooth) {
      dict->incfreq(oov,dict->size()-1);
      cerr << "Witten-Bell estimate of OOV freq:"<< (double)(dict->size()-1)/dict->totfreq() << "\n";
    } else {
      if (dict->dub()) {
        cerr << "DUB estimate of OOV size\n";
        dict->incfreq(oov,dict->dub()-dict->size()+1);
      } else {
        cerr << "1 = estimate of OOV size\n";
        dict->incfreq(oov,1);
      }
    }
  }
}


double interplm::unigr(ngram ng)
{

  return
    ((double)(dict->freq(*ng.wordp(1))+epsilon))/
    ((double)dict->totfreq() + (double) dict->size() * epsilon);

}


interplm::interplm(char *ngtfile,int depth,TABLETYPE tabtype):
  ngramtable(ngtfile,depth,NULL,NULL,NULL,0,0,NULL,0,tabtype)
{

  if (maxlevel()<depth) {
    cerr << "interplm: ngramtable size is too low\n";
    exit(1);
  }

  lms=depth;
  unitbl=NULL;
  epsilon=1.0;
  unismooth=1;
  prune_singletons=0;
  prune_top_singletons=0;

  //doing something nasty: change counter of <s>

  int BoS=dict->encode(dict->BoS());
  if (BoS != dict->oovcode()) {
    cerr << "setting counter of Begin of Sentence to 1 ..." << "\n";
    dict->freq(BoS,1);
    cerr << "start_sent: " << (char *)dict->decode(BoS) << " "
         << dict->freq(BoS) << "\n";
  }

};


void interplm::gensuccstat()
{

  ngram hg(dict);
  int s1,s2;

  cerr << "Generating successor statistics\n";


  for (int l=2; l<=lms; l++) {

    cerr << "level " << l << "\n";

    scan(hg,INIT,l-1);
    while(scan(hg,CONT,l-1)) {

      s1=s2=0;

      ngram ng=hg;
      ng.pushc(0);

      succscan(hg,ng,INIT,l);
      while(succscan(hg,ng,CONT,l)) {
        //	cerr << ng << "\n";

        if (corrcounts && l<lms) //use corrected counts!!!
          ng.freq=getfreq(ng.link,ng.pinfo,1);

        if (ng.freq==1) s1++;
        else if (ng.freq==2) s2++;
      }

      succ2(hg.link,s2);
      succ1(hg.link,s1);
    }
  }
}


void interplm::gencorrcounts()
{

  cerr << "Generating corrected n-gram tables\n";

  for (int l=lms-1; l>=1; l--) {

    cerr << "level " << l << "\n";

    ngram ng(dict);
    int count=0;

    //now update counts
    scan(ng,INIT,l+1);
    while(scan(ng,CONT,l+1)) {

      ngram ng2=ng;
      ng2.size--;
      if (get(ng2,ng2.size,ng2.size)) {

        if (!ng2.containsWord(dict->BoS(),1))
          //counts number of different n-grams
          setfreq(ng2.link,ng2.pinfo,1+getfreq(ng2.link,ng2.pinfo,1),1);
        else
          // use correct count for n-gram "<s> w .. .. "
          //setfreq(ng2.link,ng2.pinfo,ng2.freq+getfreq(ng2.link,ng2.pinfo,1),1);
          setfreq(ng2.link,ng2.pinfo,ng2.freq,1);
      } else {
        assert(lms==l+1);
        cerr << "cannot find2 " << ng2 << "count " << count << "\n";
        cerr << "inserting ngram and starting from scratch\n";
        ng2.pushw(dict->BoS());
        ng2.freq=100;
        put(ng2);

        cerr << "reset all counts at last level\n";

        scan(ng2,INIT,lms-1);
        while(scan(ng2,CONT,lms-1)) {
          setfreq(ng2.link,ng2.pinfo,0,1);
        }

        gencorrcounts();
        return;
      }
    }
  }

  cerr << "Updating history counts\n";

  for (int l=lms-2; l>=1; l--) {

    cerr << "level " << l << "\n";

    cerr << "reset counts\n";

    ngram ng(dict);
    scan(ng,INIT,l);
    while(scan(ng,CONT,l)) {
      freq(ng.link,ng.pinfo,0);
    }

    scan(ng,INIT,l+1);
    while(scan(ng,CONT,l+1)) {

      ngram ng2=ng;
      get(ng2,l+1,l);
      freq(ng2.link,ng2.pinfo,freq(ng2.link,ng2.pinfo)+getfreq(ng.link,ng.pinfo,1));
    }
  }

  cerr << "Adding unigram of OOV word if missing\n";
  ngram ng(dict,maxlevel());
  for (int i=1; i<=maxlevel(); i++)
    *ng.wordp(i)=dict->oovcode();

  if (!get(ng,lms,1)) {
    // oov is missing in the ngram-table
    // f(oov) = dictionary size (Witten Bell)
    ng.freq=dict->size();
    cerr << "adding oov unigram " << ng << "\n";
    put(ng);
    get(ng,lms,1);
    setfreq(ng.link,ng.pinfo,ng.freq,1);
  }

  cerr << "Replacing unigram of BoS \n";
  if (dict->encode(dict->BoS()) != dict->oovcode()) {
    ngram ng(dict,1);
    *ng.wordp(1)=dict->encode(dict->BoS());

    if (get(ng,1,1)) {
      ng.freq=1;  //putting Pr(<s>)=0 would create problems!!
      setfreq(ng.link,ng.pinfo,ng.freq,1);
    }
  }


  cerr << "compute unigram totfreq \n";
  int totf=0;
  scan(ng,INIT,1);
  while(scan(ng,CONT,1)) {
    totf+=getfreq(ng.link,ng.pinfo,1);
  }

  btotfreq(totf);

  corrcounts=1;


}


/*
void gencorrcounts2(){

  cerr << "Generating corrected n-gram tables\n";

  for (int l=lms-1;l>=1;l--){

    cerr << "level " << l << "\n";

    // tb[l]=new ngramtable(NULL,l,NULL,NULL);
    // tb[l]->dict=dict; //dict must be the same

    ngram ng(dict);
    int count=0;

    tb[l+1]->scan(ng,INIT,l+1);
    while(tb[l+1]->scan(ng,CONT,l+1)){

      count++;

      //generate tables according to Chen & Goodman, 1998

      // cerr << ng << "\n";

      ng.size--;

      if (!ng.containsWord(dict->BoS(),1)) ng.freq=1;
      //tb[l]->put(ng);
      //cerr << ng << "\n";
      //tb[l]->update(ng);

      ngram ng2=ng;

      if (tb[l]->get(ng2,ng2.size,ng2.size)){
      tb[l]->freq(ng2.link,ng2.info,0);
      }
      else{
        cerr << "cannot find " << ng2 << "count " << count << "\n";
	exit(1);
      }

      ng.size++;
    }

    //add unigram of OOV word if missing
    if (l==1){
      ngram oovw(dict,1);
      *oovw.wordp(1)=dict->oovcode();
      if (!tb[1]->get(oovw,1,1)){
	oovw.freq=dict->freq(dict->oovcode());
	cerr << "adding oov unigram " << oovw << "\n";
	tb[1]->put(oovw);
      }
    }
  }

  exit(1);

}
*/


double interplm::zerofreq(int lev)
{
  cerr << "Computing lambda: ...";
  ngram ng(dict);
  double N=0,N1=0;
  scan(ng,INIT,lev);
  while(scan(ng,CONT,lev)) {
    if ((lev==1) && (*ng.wordp(1)==dict->oovcode()))
      continue;
    N+=ng.freq;
    if (ng.freq==1) N1++;
  }
  cerr << (double)(N1/N) << "\n";
  return N1/N;
}


void interplm::test(char* filename,int size,int backoff,int checkpr,char* outpr)
{

  if (size>lmsize()) {
    cerr << "test: wrong ngram size\n";
    exit(1);
  }


  mfstream inp(filename,ios::in );

  char header[100];
  inp >> header;
  inp.close();

  if (strncmp(header,"nGrAm",5)==0 ||
      strncmp(header,"NgRaM",5)==0) {
    ngramtable ngt(filename,size,NULL,NULL,NULL,0,0,NULL,0,COUNT);
    test_ngt(ngt,size,backoff,checkpr);
  } else
    test_txt(filename,size,backoff,checkpr,outpr);
}


void interplm::test_txt(char* filename,int size,int /* unused parameter: backoff */,int checkpr,char* outpr)
{

  cerr << "test text " << filename << " ";
  mfstream inp(filename,ios::in );
  ngram ng(dict);

  double n=0,lp=0,pr;
  double oov=0;
  cout.precision(10);
  mfstream outp(outpr?outpr:"/dev/null",ios::out );

  if (checkpr)
    cerr << "checking probabilities\n";

  while(inp >> ng)
    if (ng.size>=1) {

      ng.size=ng.size>size?size:ng.size;

      if (dict->encode(dict->BoS()) != dict->oovcode()) {
        if (*ng.wordp(1) == dict->encode(dict->BoS())) {
          ng.size=1; //reset n-grams starting with BoS
          continue;
        }
      }

      pr=prob(ng,ng.size);

      if (outpr)
        outp << ng << "[" << ng.size << "-gram]" << " " << pr << " " << log(pr)/log(10.0) << std::endl;

      lp-=log(pr);

      n++;

      if (((int) n % 10000)==0) cerr << ".";

      if (*ng.wordp(1) == dict->oovcode()) oov++;

      if (checkpr) {
        double totp=0.0;
        int oldw=*ng.wordp(1);
        for (int c=0; c<dict->size(); c++) {
          *ng.wordp(1)=c;
          totp+=prob(ng,ng.size);
        }
        *ng.wordp(1)=oldw;

        if ( totp < (1.0 - 1e-5) ||
             totp > (1.0 + 1e-5))
          cout << ng << " " << pr << " [t="<< totp << "] ***\n";
      }

    }

  if (oov && dict->dub()>obswrd())
    lp += oov * log(dict->dub() - obswrd());

  cout << "n=" << (int) n << " LP="
       << (double) lp
       << " PP=" << exp(lp/n)
       << " OVVRate=" << (oov)/n
       //<< " OVVLEXRate=" << (oov-in_oov_list)/n
       // << " OOVPP=" << exp((lp+oovlp)/n)

       << "\n";


  outp.close();
  inp.close();
}


void interplm::test_ngt(ngramtable& ngt,int sz,int /* unused parameter: backoff */,int checkpr)
{

  double pr;
  int n=0,c=0;
  double lp=0;
  double oov=0;
  cout.precision(10);

  if (sz > ngt.maxlevel()) {
    cerr << "test_ngt: ngramtable has uncompatible size\n";
    exit(1);
  }

  if (checkpr) cerr << "checking probabilities\n";

  cerr << "Computing PP:";

  ngram ng(dict);
  ngram ng2(ngt.dict);
  ngt.scan(ng2,INIT,sz);

  while(ngt.scan(ng2,CONT,sz)) {

    ng.trans(ng2);

    if (dict->encode(dict->BoS()) != dict->oovcode()) {
      if (*ng.wordp(1) == dict->encode(dict->BoS())) {
        ng.size=1; //reset n-grams starting with BoS
        continue;
      }
    }

    n+=ng.freq;
    pr=prob(ng,sz);

    lp-=(ng.freq * log(pr));

    if (*ng.wordp(1) == dict->oovcode())
      oov+=ng.freq;


    if (checkpr) {
      double totp=0.0;
      for (c=0; c<dict->size(); c++) {
        *ng.wordp(1)=c;
        totp+=prob(ng,sz);
      }

      if ( totp < (1.0 - 1e-5) ||
           totp > (1.0 + 1e-5))
        cout << ng << " " << pr << " [t="<< totp << "] ***\n";

    }

    if ((++c % 100000)==0) cerr << ".";

  }

  //double oovlp=oov * log((double)(dict->dub() - obswrd()));


  if (oov && dict->dub()>obswrd())

    lp+=oov * log((dict->dub() - obswrd()));

  cout << "n=" << (int) n << " LP="
       << (double) lp
       << " PP=" << exp(lp/n)
       << " OVVRate=" << (oov)/n
       //<< " OVVLEXRate=" << (oov-in_oov_list)/n
       // << " OOVPP=" << exp((lp+oovlp)/n)

       << "\n";

  cout.flush();

}




/*
main(int argc, char** argv){
  dictionary d(argv[1]);

  shiftbeta ilm(&d,argv[2],3);

  ngramtable test(&d,argv[2],3);
  ilm.train();
  cerr << "PP " << ilm.test(test) << "\n";

  ilm.savebin("newlm.lm",3);
}



*/
