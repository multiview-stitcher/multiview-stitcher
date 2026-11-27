var kw=Object.create;var jc=Object.defineProperty;var Tw=Object.getOwnPropertyDescriptor;var Nw=Object.getOwnPropertyNames;var Dw=Object.getPrototypeOf,Pw=Object.prototype.hasOwnProperty;var q=(e,t)=>()=>(t||e((t={exports:{}}).exports,t),t.exports),Pi=(e,t)=>{for(var r in t)jc(e,r,{get:t[r],enumerable:!0})},Rw=(e,t,r,n)=>{if(t&&typeof t=="object"||typeof t=="function")for(let i of Nw(t))!Pw.call(e,i)&&i!==r&&jc(e,i,{get:()=>t[i],enumerable:!(n=Tw(t,i))||n.enumerable});return e};var Ri=(e,t,r)=>(r=e!=null?kw(Dw(e)):{},Rw(t||!e||!e.__esModule?jc(r,"default",{value:e,enumerable:!0}):r,e));var At=q((Gc,Kh)=>{"use strict";var Oi=function(e){return e&&e.Math===Math&&e};Kh.exports=Oi(typeof globalThis=="object"&&globalThis)||Oi(typeof window=="object"&&window)||Oi(typeof self=="object"&&self)||Oi(typeof global=="object"&&global)||Oi(typeof Gc=="object"&&Gc)||function(){return this}()||Function("return this")()});var Jh=q((h3,Yh)=>{"use strict";var Ow=At();Yh.exports=Ow});var Qr=q((p3,Hh)=>{"use strict";Hh.exports=function(e){try{return!!e()}catch{return!0}}});var qc=q((d3,Wh)=>{"use strict";var Uw=Qr();Wh.exports=!Uw(function(){var e=function(){}.bind();return typeof e!="function"||e.hasOwnProperty("prototype")})});var bn=q((m3,Qh)=>{"use strict";var Xh=qc(),Zh=Function.prototype,Kc=Zh.call,Lw=Xh&&Zh.bind.bind(Kc,Kc);Qh.exports=Xh?Lw:function(e){return function(){return Kc.apply(e,arguments)}}});var Yc=q((g3,ep)=>{"use strict";ep.exports=function(e){return e==null}});var Jc=q((y3,tp)=>{"use strict";var Fw=Yc(),Bw=TypeError;tp.exports=function(e){if(Fw(e))throw new Bw("Can't call method on "+e);return e}});var np=q((v3,rp)=>{"use strict";var zw=Jc(),$w=Object;rp.exports=function(e){return $w(zw(e))}});var $s=q((x3,ip)=>{"use strict";var Vw=bn(),jw=np(),Gw=Vw({}.hasOwnProperty);ip.exports=Object.hasOwn||function(t,r){return Gw(jw(t),r)}});var op=q((S3,sp)=>{"use strict";sp.exports=!1});var lp=q((w3,cp)=>{"use strict";var ap=At(),qw=Object.defineProperty;cp.exports=function(e,t){try{qw(ap,e,{value:t,configurable:!0,writable:!0})}catch{ap[e]=t}return t}});var pp=q((E3,hp)=>{"use strict";var Kw=op(),Yw=At(),Jw=lp(),up="__core-js_shared__",fp=hp.exports=Yw[up]||Jw(up,{});(fp.versions||(fp.versions=[])).push({version:"3.51.0",mode:Kw?"pure":"global",copyright:"\xA9 2013\u20132025 Denis Pushkarev (zloirock.ru), 2025\u20132026 CoreJS Company (core-js.io). All rights reserved.",license:"https://github.com/zloirock/core-js/blob/v3.51.0/LICENSE",source:"https://github.com/zloirock/core-js"})});var gp=q((b3,mp)=>{"use strict";var dp=pp(),Hw=Object.create||Object;mp.exports=function(e,t){return dp[e]||(dp[e]=t||Hw(null))}});var vp=q((I3,yp)=>{"use strict";var Ww=bn(),Xw=0,Zw=Math.random(),Qw=Ww(1.1.toString);yp.exports=function(e){return"Symbol("+(e===void 0?"":e)+")_"+Qw(++Xw+Zw,36)}});var Ep=q((_3,wp)=>{"use strict";var eE=At(),xp=eE.navigator,Sp=xp&&xp.userAgent;wp.exports=Sp?String(Sp):""});var kp=q((C3,Mp)=>{"use strict";var Ap=At(),Hc=Ep(),bp=Ap.process,Ip=Ap.Deno,_p=bp&&bp.versions||Ip&&Ip.version,Cp=_p&&_p.v8,Vt,Vs;Cp&&(Vt=Cp.split("."),Vs=Vt[0]>0&&Vt[0]<4?1:+(Vt[0]+Vt[1]));!Vs&&Hc&&(Vt=Hc.match(/Edge\/(\d+)/),(!Vt||Vt[1]>=74)&&(Vt=Hc.match(/Chrome\/(\d+)/),Vt&&(Vs=+Vt[1])));Mp.exports=Vs});var Wc=q((A3,Np)=>{"use strict";var Tp=kp(),tE=Qr(),rE=At(),nE=rE.String;Np.exports=!!Object.getOwnPropertySymbols&&!tE(function(){var e=Symbol("symbol detection");return!nE(e)||!(Object(e)instanceof Symbol)||!Symbol.sham&&Tp&&Tp<41})});var Xc=q((M3,Dp)=>{"use strict";var iE=Wc();Dp.exports=iE&&!Symbol.sham&&typeof Symbol.iterator=="symbol"});var Qc=q((k3,Rp)=>{"use strict";var sE=At(),oE=gp(),Pp=$s(),aE=vp(),cE=Wc(),lE=Xc(),In=sE.Symbol,Zc=oE("wks"),uE=lE?In.for||In:In&&In.withoutSetter||aE;Rp.exports=function(e){return Pp(Zc,e)||(Zc[e]=cE&&Pp(In,e)?In[e]:uE("Symbol."+e)),Zc[e]}});var js=q(Op=>{"use strict";var fE=Qc();Op.f=fE});var Ui=q((N3,Up)=>{"use strict";var hE=Qr();Up.exports=!hE(function(){return Object.defineProperty({},1,{get:function(){return 7}})[1]!==7})});var _n=q((D3,Lp)=>{"use strict";var el=typeof document=="object"&&document.all;Lp.exports=typeof el>"u"&&el!==void 0?function(e){return typeof e=="function"||e===el}:function(e){return typeof e=="function"}});var Li=q((P3,Fp)=>{"use strict";var pE=_n();Fp.exports=function(e){return typeof e=="object"?e!==null:pE(e)}});var $p=q((R3,zp)=>{"use strict";var dE=At(),Bp=Li(),tl=dE.document,mE=Bp(tl)&&Bp(tl.createElement);zp.exports=function(e){return mE?tl.createElement(e):{}}});var rl=q((O3,Vp)=>{"use strict";var gE=Ui(),yE=Qr(),vE=$p();Vp.exports=!gE&&!yE(function(){return Object.defineProperty(vE("div"),"a",{get:function(){return 7}}).a!==7})});var Gp=q((U3,jp)=>{"use strict";var xE=Ui(),SE=Qr();jp.exports=xE&&SE(function(){return Object.defineProperty(function(){},"prototype",{value:42,writable:!1}).prototype!==42})});var Kp=q((L3,qp)=>{"use strict";var wE=Li(),EE=String,bE=TypeError;qp.exports=function(e){if(wE(e))return e;throw new bE(EE(e)+" is not an object")}});var qs=q((F3,Yp)=>{"use strict";var IE=qc(),Gs=Function.prototype.call;Yp.exports=IE?Gs.bind(Gs):function(){return Gs.apply(Gs,arguments)}});var Hp=q((B3,Jp)=>{"use strict";var nl=At(),_E=_n(),CE=function(e){return _E(e)?e:void 0};Jp.exports=function(e,t){return arguments.length<2?CE(nl[e]):nl[e]&&nl[e][t]}});var Xp=q((z3,Wp)=>{"use strict";var AE=bn();Wp.exports=AE({}.isPrototypeOf)});var il=q(($3,Zp)=>{"use strict";var ME=Hp(),kE=_n(),TE=Xp(),NE=Xc(),DE=Object;Zp.exports=NE?function(e){return typeof e=="symbol"}:function(e){var t=ME("Symbol");return kE(t)&&TE(t.prototype,DE(e))}});var ed=q((V3,Qp)=>{"use strict";var PE=String;Qp.exports=function(e){try{return PE(e)}catch{return"Object"}}});var rd=q((j3,td)=>{"use strict";var RE=_n(),OE=ed(),UE=TypeError;td.exports=function(e){if(RE(e))return e;throw new UE(OE(e)+" is not a function")}});var id=q((G3,nd)=>{"use strict";var LE=rd(),FE=Yc();nd.exports=function(e,t){var r=e[t];return FE(r)?void 0:LE(r)}});var od=q((q3,sd)=>{"use strict";var sl=qs(),ol=_n(),al=Li(),BE=TypeError;sd.exports=function(e,t){var r,n;if(t==="string"&&ol(r=e.toString)&&!al(n=sl(r,e))||ol(r=e.valueOf)&&!al(n=sl(r,e))||t!=="string"&&ol(r=e.toString)&&!al(n=sl(r,e)))return n;throw new BE("Can't convert object to primitive value")}});var ud=q((K3,ld)=>{"use strict";var zE=qs(),ad=Li(),cd=il(),$E=id(),VE=od(),jE=Qc(),GE=TypeError,qE=jE("toPrimitive");ld.exports=function(e,t){if(!ad(e)||cd(e))return e;var r=$E(e,qE),n;if(r){if(t===void 0&&(t="default"),n=zE(r,e,t),!ad(n)||cd(n))return n;throw new GE("Can't convert object to primitive value")}return t===void 0&&(t="number"),VE(e,t)}});var cl=q((Y3,fd)=>{"use strict";var KE=ud(),YE=il();fd.exports=function(e){var t=KE(e,"string");return YE(t)?t:t+""}});var Ys=q(pd=>{"use strict";var JE=Ui(),HE=rl(),WE=Gp(),Ks=Kp(),hd=cl(),XE=TypeError,ll=Object.defineProperty,ZE=Object.getOwnPropertyDescriptor,ul="enumerable",fl="configurable",hl="writable";pd.f=JE?WE?function(t,r,n){if(Ks(t),r=hd(r),Ks(n),typeof t=="function"&&r==="prototype"&&"value"in n&&hl in n&&!n[hl]){var i=ZE(t,r);i&&i[hl]&&(t[r]=n.value,n={configurable:fl in n?n[fl]:i[fl],enumerable:ul in n?n[ul]:i[ul],writable:!1})}return ll(t,r,n)}:ll:function(t,r,n){if(Ks(t),r=hd(r),Ks(n),HE)try{return ll(t,r,n)}catch{}if("get"in n||"set"in n)throw new XE("Accessors not supported");return"value"in n&&(t[r]=n.value),t}});var pl=q((H3,md)=>{"use strict";var dd=Jh(),QE=$s(),eb=js(),tb=Ys().f;md.exports=function(e){var t=dd.Symbol||(dd.Symbol={});QE(t,e)||tb(t,e,{value:eb.f(e)})}});var xd=q(vd=>{"use strict";var gd={}.propertyIsEnumerable,yd=Object.getOwnPropertyDescriptor,rb=yd&&!gd.call({1:2},1);vd.f=rb?function(t){var r=yd(this,t);return!!r&&r.enumerable}:gd});var wd=q((X3,Sd)=>{"use strict";Sd.exports=function(e,t){return{enumerable:!(e&1),configurable:!(e&2),writable:!(e&4),value:t}}});var Id=q((Z3,bd)=>{"use strict";var Ed=bn(),nb=Ed({}.toString),ib=Ed("".slice);bd.exports=function(e){return ib(nb(e),8,-1)}});var Cd=q((Q3,_d)=>{"use strict";var sb=bn(),ob=Qr(),ab=Id(),dl=Object,cb=sb("".split);_d.exports=ob(function(){return!dl("z").propertyIsEnumerable(0)})?function(e){return ab(e)==="String"?cb(e,""):dl(e)}:dl});var Md=q((eP,Ad)=>{"use strict";var lb=Cd(),ub=Jc();Ad.exports=function(e){return lb(ub(e))}});var ml=q(Td=>{"use strict";var fb=Ui(),hb=qs(),pb=xd(),db=wd(),mb=Md(),gb=cl(),yb=$s(),vb=rl(),kd=Object.getOwnPropertyDescriptor;Td.f=fb?kd:function(t,r){if(t=mb(t),r=gb(r),vb)try{return kd(t,r)}catch{}if(yb(t,r))return db(!hb(pb.f,t,r),t[r])}});var yl=q(()=>{"use strict";var xb=At(),Sb=pl(),wb=Ys().f,Eb=ml().f,gl=xb.Symbol;Sb("dispose");gl&&(Fi=Eb(gl,"dispose"),Fi.enumerable&&Fi.configurable&&Fi.writable&&wb(gl,"dispose",{value:Fi.value,enumerable:!1,configurable:!1,writable:!1}));var Fi});var Dd=q((iP,Nd)=>{"use strict";yl();var bb=js();Nd.exports=bb.f("dispose")});var Rd=q((sP,Pd)=>{"use strict";var Ib=Dd();Pd.exports=Ib});var Od=q(()=>{"use strict";yl()});var Ld=q((cP,Ud)=>{"use strict";var _b=Rd();Od();Ud.exports=_b});var xl=q(()=>{"use strict";var Cb=At(),Ab=pl(),Mb=Ys().f,kb=ml().f,vl=Cb.Symbol;Ab("asyncDispose");vl&&(Bi=kb(vl,"asyncDispose"),Bi.enumerable&&Bi.configurable&&Bi.writable&&Mb(vl,"asyncDispose",{value:Bi.value,enumerable:!1,configurable:!1,writable:!1}));var Bi});var Bd=q((fP,Fd)=>{"use strict";xl();var Tb=js();Fd.exports=Tb.f("asyncDispose")});var $d=q((hP,zd)=>{"use strict";var Nb=Bd();zd.exports=Nb});var Vd=q(()=>{"use strict";xl()});var Gd=q((mP,jd)=>{"use strict";var Db=$d();Vd();jd.exports=Db});var Tx=q(H=>{"use strict";H.deflate=yx;H.deflateSync=vs;H.inflate=Rf;H.inflateSync=vi;H.gzip=qa;H.compress=qa;H.gzipSync=La;H.compressSync=La;H.gunzip=Sx;H.gunzipSync=Ba;H.zlib=Ek;H.zlibSync=bf;H.unzlib=Ex;H.unzlibSync=$a;H.gzip=qa;H.compress=qa;H.decompress=Ik;H.decompressSync=_k;H.strToU8=Yr;H.strFromU8=Uf;H.zip=Nk;H.zipSync=Dk;H.unzip=Uk;H.unzipSync=Lk;var Jv={},Xv={};Xv.default=function(e,t,r,n,i){var s=new Worker(Jv[t]||(Jv[t]=URL.createObjectURL(new Blob([e+';addEventListener("error",function(e){e=e.error;postMessage({$e$:[e.message,e.code,e.stack]})})'],{type:"text/javascript"}))));return s.onmessage=function(o){var a=o.data,c=a.$e$;if(c){var l=new Error(c[0]);l.code=c[1],l.stack=c[2],i(l,null)}else i(null,a)},s.postMessage(r,n),s};var ee=Uint8Array,nt=Uint16Array,ms=Int32Array,ui=new ee([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0]),fi=new ee([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,0,0]),hs=new ee([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]),Zv=function(e,t){for(var r=new nt(31),n=0;n<31;++n)r[n]=t+=1<<e[n-1];for(var i=new ms(r[30]),n=1;n<30;++n)for(var s=r[n];s<r[n+1];++s)i[s]=s-r[n]<<5|n;return{b:r,r:i}},Qv=Zv(ui,2),Cf=Qv.b,Ra=Qv.r;Cf[28]=258,Ra[258]=28;var ex=Zv(fi,0),tx=ex.b,xf=ex.r,ps=new nt(32768);for(pe=0;pe<32768;++pe)Ir=(pe&43690)>>1|(pe&21845)<<1,Ir=(Ir&52428)>>2|(Ir&13107)<<2,Ir=(Ir&61680)>>4|(Ir&3855)<<4,ps[pe]=((Ir&65280)>>8|(Ir&255)<<8)>>1;var Ir,pe,Rt=function(e,t,r){for(var n=e.length,i=0,s=new nt(t);i<n;++i)e[i]&&++s[e[i]-1];var o=new nt(t);for(i=1;i<t;++i)o[i]=o[i-1]+s[i-1]<<1;var a;if(r){a=new nt(1<<t);var c=15-t;for(i=0;i<n;++i)if(e[i])for(var l=i<<4|e[i],u=t-e[i],f=o[e[i]-1]++<<u,h=f|(1<<u)-1;f<=h;++f)a[ps[f]>>c]=l}else for(a=new nt(n),i=0;i<n;++i)e[i]&&(a[i]=ps[o[e[i]-1]++]>>15-e[i]);return a},_r=new ee(288);for(pe=0;pe<144;++pe)_r[pe]=8;var pe;for(pe=144;pe<256;++pe)_r[pe]=9;var pe;for(pe=256;pe<280;++pe)_r[pe]=7;var pe;for(pe=280;pe<288;++pe)_r[pe]=8;var pe,ci=new ee(32);for(pe=0;pe<32;++pe)ci[pe]=5;var pe,rx=Rt(_r,9,0),nx=Rt(_r,9,1),ix=Rt(ci,5,0),sx=Rt(ci,5,1),Na=function(e){for(var t=e[0],r=1;r<e.length;++r)e[r]>t&&(t=e[r]);return t},Pt=function(e,t,r){var n=t/8|0;return(e[n]|e[n+1]<<8)>>(t&7)&r},Da=function(e,t){var r=t/8|0;return(e[r]|e[r+1]<<8|e[r+2]<<16)>>(t&7)},hi=function(e){return(e+7)/8|0},Ot=function(e,t,r){return(t==null||t<0)&&(t=0),(r==null||r>e.length)&&(r=e.length),new ee(e.subarray(t,r))};H.FlateErrorCode={UnexpectedEOF:0,InvalidBlockType:1,InvalidLengthLiteral:2,InvalidDistance:3,StreamFinished:4,NoStreamHandler:5,InvalidHeader:6,NoCallback:7,InvalidUTF8:8,ExtraFieldTooLong:9,InvalidDate:10,FilenameTooLong:11,StreamFinishing:12,InvalidZipData:13,UnknownCompressionMethod:14};var ox=["unexpected EOF","invalid block type","invalid length/literal","invalid distance","stream finished","no stream handler",,"no callback","invalid UTF-8 data","extra field too long","date not in range 1980-2099","filename too long","stream finishing","invalid zip data"],K=function(e,t,r){var n=new Error(t||ox[e]);if(n.code=e,Error.captureStackTrace&&Error.captureStackTrace(n,K),!r)throw n;return n},gs=function(e,t,r,n){var i=e.length,s=n?n.length:0;if(!i||t.f&&!t.l)return r||new ee(0);var o=!r,a=o||t.i!=2,c=t.i;o&&(r=new ee(i*3));var l=function(vt){var dr=r.length;if(vt>dr){var kr=new ee(Math.max(dr*2,vt));kr.set(r),r=kr}},u=t.f||0,f=t.p||0,h=t.b||0,p=t.l,d=t.d,m=t.m,g=t.n,y=i*8;do{if(!p){u=Pt(e,f,1);var I=Pt(e,f+1,3);if(f+=3,I)if(I==1)p=nx,d=sx,m=9,g=5;else if(I==2){var C=Pt(e,f,31)+257,v=Pt(e,f+10,15)+4,w=C+Pt(e,f+5,31)+1;f+=14;for(var x=new ee(w),k=new ee(19),M=0;M<v;++M)k[hs[M]]=Pt(e,f+M*3,7);f+=v*3;for(var P=Na(k),F=(1<<P)-1,S=Rt(k,P,1),M=0;M<w;){var O=S[Pt(e,f,F)];f+=O&15;var _=O>>4;if(_<16)x[M++]=_;else{var R=0,N=0;for(_==16?(N=3+Pt(e,f,3),f+=2,R=x[M-1]):_==17?(N=3+Pt(e,f,7),f+=3):_==18&&(N=11+Pt(e,f,127),f+=7);N--;)x[M++]=R}}var U=x.subarray(0,C),L=x.subarray(C);m=Na(U),g=Na(L),p=Rt(U,m,1),d=Rt(L,g,1)}else K(1);else{var _=hi(f)+4,E=e[_-4]|e[_-3]<<8,b=_+E;if(b>i){c&&K(0);break}a&&l(h+E),r.set(e.subarray(_,b),h),t.b=h+=E,t.p=f=b*8,t.f=u;continue}if(f>y){c&&K(0);break}}a&&l(h+131072);for(var G=(1<<m)-1,V=(1<<g)-1,Y=f;;Y=f){var R=p[Da(e,f)&G],J=R>>4;if(f+=R&15,f>y){c&&K(0);break}if(R||K(2),J<256)r[h++]=J;else if(J==256){Y=f,p=null;break}else{var de=J-254;if(J>264){var M=J-257,ie=ui[M];de=Pt(e,f,(1<<ie)-1)+Cf[M],f+=ie}var Ce=d[Da(e,f)&V],ze=Ce>>4;Ce||K(3),f+=Ce&15;var L=tx[ze];if(ze>3){var ie=fi[ze];L+=Da(e,f)&(1<<ie)-1,f+=ie}if(f>y){c&&K(0);break}a&&l(h+131072);var at=h+de;if(h<L){var gt=s-L,yt=Math.min(L,at);for(gt+h<0&&K(3);h<yt;++h)r[h]=n[gt+h]}for(;h<at;++h)r[h]=r[h-L]}}t.l=p,t.p=Y,t.b=h,t.f=u,p&&(u=1,t.m=m,t.d=d,t.n=g)}while(!u);return h!=r.length&&o?Ot(r,0,h):r.subarray(0,h)},ar=function(e,t,r){r<<=t&7;var n=t/8|0;e[n]|=r,e[n+1]|=r>>8},oi=function(e,t,r){r<<=t&7;var n=t/8|0;e[n]|=r,e[n+1]|=r>>8,e[n+2]|=r>>16},Pa=function(e,t){for(var r=[],n=0;n<e.length;++n)e[n]&&r.push({s:n,f:e[n]});var i=r.length,s=r.slice();if(!i)return{t:cr,l:0};if(i==1){var o=new ee(r[0].s+1);return o[r[0].s]=1,{t:o,l:1}}r.sort(function(b,C){return b.f-C.f}),r.push({s:-1,f:25001});var a=r[0],c=r[1],l=0,u=1,f=2;for(r[0]={s:-1,f:a.f+c.f,l:a,r:c};u!=i-1;)a=r[r[l].f<r[f].f?l++:f++],c=r[l!=u&&r[l].f<r[f].f?l++:f++],r[u++]={s:-1,f:a.f+c.f,l:a,r:c};for(var h=s[0].s,n=1;n<i;++n)s[n].s>h&&(h=s[n].s);var p=new nt(h+1),d=Oa(r[u-1],p,0);if(d>t){var n=0,m=0,g=d-t,y=1<<g;for(s.sort(function(C,v){return p[v.s]-p[C.s]||C.f-v.f});n<i;++n){var I=s[n].s;if(p[I]>t)m+=y-(1<<d-p[I]),p[I]=t;else break}for(m>>=g;m>0;){var _=s[n].s;p[_]<t?m-=1<<t-p[_]++-1:++n}for(;n>=0&&m;--n){var E=s[n].s;p[E]==t&&(--p[E],++m)}d=t}return{t:new ee(p),l:d}},Oa=function(e,t,r){return e.s==-1?Math.max(Oa(e.l,t,r+1),Oa(e.r,t,r+1)):t[e.s]=r},Sf=function(e){for(var t=e.length;t&&!e[--t];);for(var r=new nt(++t),n=0,i=e[0],s=1,o=function(c){r[n++]=c},a=1;a<=t;++a)if(e[a]==i&&a!=t)++s;else{if(!i&&s>2){for(;s>138;s-=138)o(32754);s>2&&(o(s>10?s-11<<5|28690:s-3<<5|12305),s=0)}else if(s>3){for(o(i),--s;s>6;s-=6)o(8304);s>2&&(o(s-3<<5|8208),s=0)}for(;s--;)o(i);s=1,i=e[a]}return{c:r.subarray(0,n),n:t}},ai=function(e,t){for(var r=0,n=0;n<t.length;++n)r+=e[n]*t[n];return r},ja=function(e,t,r){var n=r.length,i=hi(t+2);e[i]=n&255,e[i+1]=n>>8,e[i+2]=e[i]^255,e[i+3]=e[i+1]^255;for(var s=0;s<n;++s)e[i+s+4]=r[s];return(i+4+n)*8},wf=function(e,t,r,n,i,s,o,a,c,l,u){ar(t,u++,r),++i[256];for(var f=Pa(i,15),h=f.t,p=f.l,d=Pa(s,15),m=d.t,g=d.l,y=Sf(h),I=y.c,_=y.n,E=Sf(m),b=E.c,C=E.n,v=new nt(19),w=0;w<I.length;++w)++v[I[w]&31];for(var w=0;w<b.length;++w)++v[b[w]&31];for(var x=Pa(v,7),k=x.t,M=x.l,P=19;P>4&&!k[hs[P-1]];--P);var F=l+5<<3,S=ai(i,_r)+ai(s,ci)+o,O=ai(i,h)+ai(s,m)+o+14+3*P+ai(v,k)+2*v[16]+3*v[17]+7*v[18];if(c>=0&&F<=S&&F<=O)return ja(t,u,e.subarray(c,c+l));var R,N,U,L;if(ar(t,u,1+(O<S)),u+=2,O<S){R=Rt(h,p,0),N=h,U=Rt(m,g,0),L=m;var G=Rt(k,M,0);ar(t,u,_-257),ar(t,u+5,C-1),ar(t,u+10,P-4),u+=14;for(var w=0;w<P;++w)ar(t,u+3*w,k[hs[w]]);u+=3*P;for(var V=[I,b],Y=0;Y<2;++Y)for(var J=V[Y],w=0;w<J.length;++w){var de=J[w]&31;ar(t,u,G[de]),u+=k[de],de>15&&(ar(t,u,J[w]>>5&127),u+=J[w]>>12)}}else R=rx,N=_r,U=ix,L=ci;for(var w=0;w<a;++w){var ie=n[w];if(ie>255){var de=ie>>18&31;oi(t,u,R[de+257]),u+=N[de+257],de>7&&(ar(t,u,ie>>23&31),u+=ui[de]);var Ce=ie&31;oi(t,u,U[Ce]),u+=L[Ce],Ce>3&&(oi(t,u,ie>>5&8191),u+=fi[Ce])}else oi(t,u,R[ie]),u+=N[ie]}return oi(t,u,R[256]),u+N[256]},ax=new ms([65540,131080,131088,131104,262176,1048704,1048832,2114560,2117632]),cr=new ee(0),cx=function(e,t,r,n,i,s){var o=s.z||e.length,a=new ee(n+o+5*(1+Math.ceil(o/7e3))+i),c=a.subarray(n,a.length-i),l=s.l,u=(s.r||0)&7;if(t){u&&(c[0]=s.r>>3);for(var f=ax[t-1],h=f>>13,p=f&8191,d=(1<<r)-1,m=s.p||new nt(32768),g=s.h||new nt(d+1),y=Math.ceil(r/3),I=2*y,_=function(ki){return(e[ki]^e[ki+1]<<y^e[ki+2]<<I)&d},E=new ms(25e3),b=new nt(288),C=new nt(32),v=0,w=0,x=s.i||0,k=0,M=s.w||0,P=0;x+2<o;++x){var F=_(x),S=x&32767,O=g[F];if(m[S]=O,g[F]=S,M<=x){var R=o-x;if((v>7e3||k>24576)&&(R>423||!l)){u=wf(e,c,0,E,b,C,w,k,P,x-P,u),k=v=w=0,P=x;for(var N=0;N<286;++N)b[N]=0;for(var N=0;N<30;++N)C[N]=0}var U=2,L=0,G=p,V=S-O&32767;if(R>2&&F==_(x-V))for(var Y=Math.min(h,R)-1,J=Math.min(32767,x),de=Math.min(258,R);V<=J&&--G&&S!=O;){if(e[x+U]==e[x+U-V]){for(var ie=0;ie<de&&e[x+ie]==e[x+ie-V];++ie);if(ie>U){if(U=ie,L=V,ie>Y)break;for(var Ce=Math.min(V,ie-2),ze=0,N=0;N<Ce;++N){var at=x-V+N&32767,gt=m[at],yt=at-gt&32767;yt>ze&&(ze=yt,O=at)}}}S=O,O=m[S],V+=S-O&32767}if(L){E[k++]=268435456|Ra[U]<<18|xf[L];var vt=Ra[U]&31,dr=xf[L]&31;w+=ui[vt]+fi[dr],++b[257+vt],++C[dr],M=x+U,++v}else E[k++]=e[x],++b[e[x]]}}for(x=Math.max(x,M);x<o;++x)E[k++]=e[x],++b[e[x]];u=wf(e,c,l,E,b,C,w,k,P,x-P,u),l||(s.r=u&7|c[u/8|0]<<3,u-=7,s.h=g,s.p=m,s.i=x,s.w=M)}else{for(var x=s.w||0;x<o+l;x+=65535){var kr=x+65535;kr>=o&&(c[u/8|0]=l,kr=o),u=ja(c,u+1,e.subarray(x,kr))}s.i=o}return Ot(a,0,n+hi(u)+i)},lx=function(){for(var e=new Int32Array(256),t=0;t<256;++t){for(var r=t,n=9;--n;)r=(r&1&&-306674912)^r>>>1;e[t]=r}return e}(),pi=function(){var e=-1;return{p:function(t){for(var r=e,n=0;n<t.length;++n)r=lx[r&255^t[n]]^r>>>8;e=r},d:function(){return~e}}},Ga=function(){var e=1,t=0;return{p:function(r){for(var n=e,i=t,s=r.length|0,o=0;o!=s;){for(var a=Math.min(o+2655,s);o<a;++o)i+=n+=r[o];n=(n&65535)+15*(n>>16),i=(i&65535)+15*(i>>16)}e=n,t=i},d:function(){return e%=65521,t%=65521,(e&255)<<24|(e&65280)<<8|(t&255)<<8|t>>8}}},gn=function(e,t,r,n,i){if(!i&&(i={l:1},t.dictionary)){var s=t.dictionary.subarray(-32768),o=new ee(s.length+e.length);o.set(s),o.set(e,s.length),e=o,i.w=s.length}return cx(e,t.level==null?6:t.level,t.mem==null?i.l?Math.ceil(Math.max(8,Math.min(13,Math.log(e.length)))*1.5):20:12+t.mem,r,n,i)},ys=function(e,t){var r={};for(var n in e)r[n]=e[n];for(var n in t)r[n]=t[n];return r},Hv=function(e,t,r){for(var n=e(),i=e.toString(),s=i.slice(i.indexOf("[")+1,i.lastIndexOf("]")).replace(/\s+/g,"").split(","),o=0;o<n.length;++o){var a=n[o],c=s[o];if(typeof a=="function"){t+=";"+c+"=";var l=a.toString();if(a.prototype)if(l.indexOf("[native code]")!=-1){var u=l.indexOf(" ",8)+1;t+=l.slice(u,l.indexOf("(",u))}else{t+=l;for(var f in a.prototype)t+=";"+c+".prototype."+f+"="+a.prototype[f].toString()}else t+=l}else r[c]=a}return t},Ta=[],Sk=function(e){var t=[];for(var r in e)e[r].buffer&&t.push((e[r]=new e[r].constructor(e[r])).buffer);return t},ux=function(e,t,r,n){if(!Ta[r]){for(var i="",s={},o=e.length-1,a=0;a<o;++a)i=Hv(e[a],i,s);Ta[r]={c:Hv(e[o],i,s),e:s}}var c=ys({},Ta[r].e);return(0,Xv.default)(Ta[r].c+";onmessage=function(e){for(var k in e.data)self[k]=e.data[k];onmessage="+t.toString()+"}",r,c,Sk(c),n)},di=function(){return[ee,nt,ms,ui,fi,hs,Cf,tx,nx,sx,ps,ox,Rt,Na,Pt,Da,hi,Ot,K,gs,vi,Jr,Af]},mi=function(){return[ee,nt,ms,ui,fi,hs,Ra,xf,rx,_r,ix,ci,ps,ax,cr,Rt,ar,oi,Pa,Oa,Sf,ai,ja,wf,hi,Ot,cx,gn,vs,Jr]},fx=function(){return[Mf,Tf,he,pi,lx]},hx=function(){return[kf,mx]},px=function(){return[Nf,he,Ga]},dx=function(){return[Df]},Jr=function(e){return postMessage(e,[e.buffer])},Af=function(e){return e&&{out:e.size&&new ee(e.size),dictionary:e.dictionary}},gi=function(e,t,r,n,i,s){var o=ux(r,n,i,function(a,c){o.terminate(),s(a,c)});return o.postMessage([e,t],t.consume?[e.buffer]:[]),function(){o.terminate()}},Lt=function(e){return e.ondata=function(t,r){return postMessage([t,r],[t.buffer])},function(t){t.data[0]?(e.push(t.data[0],t.data[1]),postMessage([t.data[0].length])):e.flush(t.data[1])}},yi=function(e,t,r,n,i,s,o){var a,c=ux(e,n,i,function(l,u){l?(c.terminate(),t.ondata.call(t,l)):Array.isArray(u)?u.length==1?(t.queuedSize-=u[0],t.ondrain&&t.ondrain(u[0])):(u[1]&&c.terminate(),t.ondata.call(t,l,u[0],u[1])):o(u)});c.postMessage(r),t.queuedSize=0,t.push=function(l,u){t.ondata||K(5),a&&t.ondata(K(4,0,1),null,!!u),t.queuedSize+=l.length,c.postMessage([l,a=u],l.buffer instanceof ArrayBuffer?[l.buffer]:[])},t.terminate=function(){c.terminate()},s&&(t.flush=function(l){c.postMessage([0,l])})},rt=function(e,t){return e[t]|e[t+1]<<8},Ie=function(e,t){return(e[t]|e[t+1]<<8|e[t+2]<<16|e[t+3]<<24)>>>0},vf=function(e,t){return Ie(e,t)+Ie(e,t+4)*4294967296},he=function(e,t,r){for(;r;++t)e[t]=r,r>>>=8},Mf=function(e,t){var r=t.filename;if(e[0]=31,e[1]=139,e[2]=8,e[8]=t.level<2?4:t.level==9?2:0,e[9]=3,t.mtime!=0&&he(e,4,Math.floor(new Date(t.mtime||Date.now())/1e3)),r){e[3]=8;for(var n=0;n<=r.length;++n)e[n+10]=r.charCodeAt(n)}},kf=function(e){(e[0]!=31||e[1]!=139||e[2]!=8)&&K(6,"invalid gzip data");var t=e[3],r=10;t&4&&(r+=(e[10]|e[11]<<8)+2);for(var n=(t>>3&1)+(t>>4&1);n>0;n-=!e[r++]);return r+(t&2)},mx=function(e){var t=e.length;return(e[t-4]|e[t-3]<<8|e[t-2]<<16|e[t-1]<<24)>>>0},Tf=function(e){return 10+(e.filename?e.filename.length+1:0)},Nf=function(e,t){var r=t.level,n=r==0?0:r<6?1:r==9?3:2;if(e[0]=120,e[1]=n<<6|(t.dictionary&&32),e[1]|=31-(e[0]<<8|e[1])%31,t.dictionary){var i=Ga();i.p(t.dictionary),he(e,2,i.d())}},Df=function(e,t){return((e[0]&15)!=8||e[0]>>4>7||(e[0]<<8|e[1])%31)&&K(6,"invalid zlib data"),(e[1]>>5&1)==+!t&&K(6,"invalid zlib data: "+(e[1]&32?"need":"unexpected")+" dictionary"),(e[1]>>3&4)+2};function yn(e,t){return typeof e=="function"&&(t=e,e={}),this.ondata=t,e}var Ut=function(){function e(t,r){if(typeof t=="function"&&(r=t,t={}),this.ondata=r,this.o=t||{},this.s={l:0,i:32768,w:32768,z:32768},this.b=new ee(98304),this.o.dictionary){var n=this.o.dictionary.subarray(-32768);this.b.set(n,32768-n.length),this.s.i=32768-n.length}}return e.prototype.p=function(t,r){this.ondata(gn(t,this.o,0,0,this.s),r)},e.prototype.push=function(t,r){this.ondata||K(5),this.s.l&&K(4);var n=t.length+this.s.z;if(n>this.b.length){if(n>2*this.b.length-32768){var i=new ee(n&-32768);i.set(this.b.subarray(0,this.s.z)),this.b=i}var s=this.b.length-this.s.z;this.b.set(t.subarray(0,s),this.s.z),this.s.z=this.b.length,this.p(this.b,!1),this.b.set(this.b.subarray(-32768)),this.b.set(t.subarray(s),32768),this.s.z=t.length-s+32768,this.s.i=32766,this.s.w=32768}else this.b.set(t,this.s.z),this.s.z+=t.length;this.s.l=r&1,(this.s.z>this.s.w+8191||r)&&(this.p(this.b,r||!1),this.s.w=this.s.i,this.s.i-=2),r&&(this.s=this.o={},this.b=cr)},e.prototype.flush=function(t){if(this.ondata||K(5),this.s.l&&K(4),this.p(this.b,!1),this.s.w=this.s.i,this.s.i-=2,t){var r=new ee(6);r[0]=this.s.r>>3;var n=ja(r,this.s.r,cr);this.s.r=0,this.ondata(r.subarray(0,n>>3),!1)}},e}();H.Deflate=Ut;var gx=function(){function e(t,r){yi([mi,function(){return[Lt,Ut]}],this,yn.call(this,t,r),function(n){var i=new Ut(n.data);onmessage=Lt(i)},6,1)}return e}();H.AsyncDeflate=gx;function yx(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[mi],function(n){return Jr(vs(n.data[0],n.data[1]))},0,r)}function vs(e,t){return gn(e,t||{},0,0)}var it=function(){function e(t,r){typeof t=="function"&&(r=t,t={}),this.ondata=r;var n=t&&t.dictionary&&t.dictionary.subarray(-32768);this.s={i:0,b:n?n.length:0},this.o=new ee(32768),this.p=new ee(0),n&&this.o.set(n)}return e.prototype.e=function(t){if(this.ondata||K(5),this.d&&K(4),!this.p.length)this.p=t;else if(t.length){var r=new ee(this.p.length+t.length);r.set(this.p),r.set(t,this.p.length),this.p=r}},e.prototype.c=function(t){this.s.i=+(this.d=t||!1);var r=this.s.b,n=gs(this.p,this.s,this.o);this.ondata(Ot(n,r,this.s.b),this.d),this.o=Ot(n,this.s.b-32768),this.s.b=this.o.length,this.p=Ot(this.p,this.s.p/8|0),this.s.p&=7},e.prototype.push=function(t,r){this.e(t),this.c(r)},e}();H.Inflate=it;var Pf=function(){function e(t,r){yi([di,function(){return[Lt,it]}],this,yn.call(this,t,r),function(n){var i=new it(n.data);onmessage=Lt(i)},7,0)}return e}();H.AsyncInflate=Pf;function Rf(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[di],function(n){return Jr(vi(n.data[0],Af(n.data[1])))},1,r)}function vi(e,t){return gs(e,{i:2},t&&t.out,t&&t.dictionary)}var Ua=function(){function e(t,r){this.c=pi(),this.l=0,this.v=1,Ut.call(this,t,r)}return e.prototype.push=function(t,r){this.c.p(t),this.l+=t.length,Ut.prototype.push.call(this,t,r)},e.prototype.p=function(t,r){var n=gn(t,this.o,this.v&&Tf(this.o),r&&8,this.s);this.v&&(Mf(n,this.o),this.v=0),r&&(he(n,n.length-8,this.c.d()),he(n,n.length-4,this.l)),this.ondata(n,r)},e.prototype.flush=function(t){Ut.prototype.flush.call(this,t)},e}();H.Gzip=Ua;H.Compress=Ua;var vx=function(){function e(t,r){yi([mi,fx,function(){return[Lt,Ut,Ua]}],this,yn.call(this,t,r),function(n){var i=new Ua(n.data);onmessage=Lt(i)},8,1)}return e}();H.AsyncGzip=vx;H.AsyncCompress=vx;function qa(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[mi,fx,function(){return[La]}],function(n){return Jr(La(n.data[0],n.data[1]))},2,r)}function La(e,t){t||(t={});var r=pi(),n=e.length;r.p(e);var i=gn(e,t,Tf(t),8),s=i.length;return Mf(i,t),he(i,s-8,r.d()),he(i,s-4,n),i}var Fa=function(){function e(t,r){this.v=1,this.r=0,it.call(this,t,r)}return e.prototype.push=function(t,r){if(it.prototype.e.call(this,t),this.r+=t.length,this.v){var n=this.p.subarray(this.v-1),i=n.length>3?kf(n):4;if(i>n.length){if(!r)return}else this.v>1&&this.onmember&&this.onmember(this.r-n.length);this.p=n.subarray(i),this.v=0}it.prototype.c.call(this,0),this.s.f&&!this.s.l?(this.v=hi(this.s.p)+9,this.s={i:0},this.o=new ee(0),this.push(new ee(0),r)):r&&it.prototype.c.call(this,r)},e}();H.Gunzip=Fa;var xx=function(){function e(t,r){var n=this;yi([di,hx,function(){return[Lt,it,Fa]}],this,yn.call(this,t,r),function(i){var s=new Fa(i.data);s.onmember=function(o){return postMessage(o)},onmessage=Lt(s)},9,0,function(i){return n.onmember&&n.onmember(i)})}return e}();H.AsyncGunzip=xx;function Sx(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[di,hx,function(){return[Ba]}],function(n){return Jr(Ba(n.data[0],n.data[1]))},3,r)}function Ba(e,t){var r=kf(e);return r+8>e.length&&K(6,"invalid gzip data"),gs(e.subarray(r,-8),{i:2},t&&t.out||new ee(mx(e)),t&&t.dictionary)}var Ef=function(){function e(t,r){this.c=Ga(),this.v=1,Ut.call(this,t,r)}return e.prototype.push=function(t,r){this.c.p(t),Ut.prototype.push.call(this,t,r)},e.prototype.p=function(t,r){var n=gn(t,this.o,this.v&&(this.o.dictionary?6:2),r&&4,this.s);this.v&&(Nf(n,this.o),this.v=0),r&&he(n,n.length-4,this.c.d()),this.ondata(n,r)},e.prototype.flush=function(t){Ut.prototype.flush.call(this,t)},e}();H.Zlib=Ef;var wk=function(){function e(t,r){yi([mi,px,function(){return[Lt,Ut,Ef]}],this,yn.call(this,t,r),function(n){var i=new Ef(n.data);onmessage=Lt(i)},10,1)}return e}();H.AsyncZlib=wk;function Ek(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[mi,px,function(){return[bf]}],function(n){return Jr(bf(n.data[0],n.data[1]))},4,r)}function bf(e,t){t||(t={});var r=Ga();r.p(e);var n=gn(e,t,t.dictionary?6:2,4);return Nf(n,t),he(n,n.length-4,r.d()),n}var za=function(){function e(t,r){it.call(this,t,r),this.v=t&&t.dictionary?2:1}return e.prototype.push=function(t,r){if(it.prototype.e.call(this,t),this.v){if(this.p.length<6&&!r)return;this.p=this.p.subarray(Df(this.p,this.v-1)),this.v=0}r&&(this.p.length<4&&K(6,"invalid zlib data"),this.p=this.p.subarray(0,-4)),it.prototype.c.call(this,r)},e}();H.Unzlib=za;var wx=function(){function e(t,r){yi([di,dx,function(){return[Lt,it,za]}],this,yn.call(this,t,r),function(n){var i=new za(n.data);onmessage=Lt(i)},11,0)}return e}();H.AsyncUnzlib=wx;function Ex(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[di,dx,function(){return[$a]}],function(n){return Jr($a(n.data[0],Af(n.data[1])))},5,r)}function $a(e,t){return gs(e.subarray(Df(e,t&&t.dictionary),-4),{i:2},t&&t.out,t&&t.dictionary)}var If=function(){function e(t,r){this.o=yn.call(this,t,r)||{},this.G=Fa,this.I=it,this.Z=za}return e.prototype.i=function(){var t=this;this.s.ondata=function(r,n){t.ondata(r,n)}},e.prototype.push=function(t,r){if(this.ondata||K(5),this.s)this.s.push(t,r);else{if(this.p&&this.p.length){var n=new ee(this.p.length+t.length);n.set(this.p),n.set(t,this.p.length)}else this.p=t;this.p.length>2&&(this.s=this.p[0]==31&&this.p[1]==139&&this.p[2]==8?new this.G(this.o):(this.p[0]&15)!=8||this.p[0]>>4>7||(this.p[0]<<8|this.p[1])%31?new this.I(this.o):new this.Z(this.o),this.i(),this.s.push(this.p,r),this.p=null)}},e}();H.Decompress=If;var bk=function(){function e(t,r){If.call(this,t,r),this.queuedSize=0,this.G=xx,this.I=Pf,this.Z=wx}return e.prototype.i=function(){var t=this;this.s.ondata=function(r,n,i){t.ondata(r,n,i)},this.s.ondrain=function(r){t.queuedSize-=r,t.ondrain&&t.ondrain(r)}},e.prototype.push=function(t,r){this.queuedSize+=t.length,If.prototype.push.call(this,t,r)},e}();H.AsyncDecompress=bk;function Ik(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),e[0]==31&&e[1]==139&&e[2]==8?Sx(e,t,r):(e[0]&15)!=8||e[0]>>4>7||(e[0]<<8|e[1])%31?Rf(e,t,r):Ex(e,t,r)}function _k(e,t){return e[0]==31&&e[1]==139&&e[2]==8?Ba(e,t):(e[0]&15)!=8||e[0]>>4>7||(e[0]<<8|e[1])%31?vi(e,t):$a(e,t)}var Of=function(e,t,r,n){for(var i in e){var s=e[i],o=t+i,a=n;Array.isArray(s)&&(a=ys(n,s[1]),s=s[0]),ArrayBuffer.isView(s)?r[o]=[s,a]:(r[o+="/"]=[new ee(0),a],Of(s,o,r,n))}},Wv=typeof TextEncoder<"u"&&new TextEncoder,_f=typeof TextDecoder<"u"&&new TextDecoder,bx=0;try{_f.decode(cr,{stream:!0}),bx=1}catch{}var Ix=function(e){for(var t="",r=0;;){var n=e[r++],i=(n>127)+(n>223)+(n>239);if(r+i>e.length)return{s:t,r:Ot(e,r-1)};i?i==3?(n=((n&15)<<18|(e[r++]&63)<<12|(e[r++]&63)<<6|e[r++]&63)-65536,t+=String.fromCharCode(55296|n>>10,56320|n&1023)):i&1?t+=String.fromCharCode((n&31)<<6|e[r++]&63):t+=String.fromCharCode((n&15)<<12|(e[r++]&63)<<6|e[r++]&63):t+=String.fromCharCode(n)}},Ck=function(){function e(t){this.ondata=t,bx?this.t=new TextDecoder:this.p=cr}return e.prototype.push=function(t,r){if(this.ondata||K(5),r=!!r,this.t){this.ondata(this.t.decode(t,{stream:!0}),r),r&&(this.t.decode().length&&K(8),this.t=null);return}this.p||K(4);var n=new ee(this.p.length+t.length);n.set(this.p),n.set(t,this.p.length);var i=Ix(n),s=i.s,o=i.r;r?(o.length&&K(8),this.p=null):this.p=o,this.ondata(s,r)},e}();H.DecodeUTF8=Ck;var Ak=function(){function e(t){this.ondata=t}return e.prototype.push=function(t,r){this.ondata||K(5),this.d&&K(4),this.ondata(Yr(t),this.d=r||!1)},e}();H.EncodeUTF8=Ak;function Yr(e,t){if(t){for(var r=new ee(e.length),n=0;n<e.length;++n)r[n]=e.charCodeAt(n);return r}if(Wv)return Wv.encode(e);for(var i=e.length,s=new ee(e.length+(e.length>>1)),o=0,a=function(u){s[o++]=u},n=0;n<i;++n){if(o+5>s.length){var c=new ee(o+8+(i-n<<1));c.set(s),s=c}var l=e.charCodeAt(n);l<128||t?a(l):l<2048?(a(192|l>>6),a(128|l&63)):l>55295&&l<57344?(l=65536+(l&1047552)|e.charCodeAt(++n)&1023,a(240|l>>18),a(128|l>>12&63),a(128|l>>6&63),a(128|l&63)):(a(224|l>>12),a(128|l>>6&63),a(128|l&63))}return Ot(s,0,o)}function Uf(e,t){if(t){for(var r="",n=0;n<e.length;n+=16384)r+=String.fromCharCode.apply(null,e.subarray(n,n+16384));return r}else{if(_f)return _f.decode(e);var i=Ix(e),s=i.s,r=i.r;return r.length&&K(8),s}}var _x=function(e){return e==1?3:e<6?2:e==9?1:0},Cx=function(e,t){return t+30+rt(e,t+26)+rt(e,t+28)},Ax=function(e,t,r){var n=rt(e,t+28),i=rt(e,t+30),s=Uf(e.subarray(t+46,t+46+n),!(rt(e,t+8)&2048)),o=t+46+n,a=Mx(e,o,i,r,Ie(e,t+20),Ie(e,t+24),Ie(e,t+42)),c=a[0],l=a[1],u=a[2];return[rt(e,t+10),c,l,s,o+i+rt(e,t+32),u]},Mx=function(e,t,r,n,i,s,o){var a=i==4294967295,c=s==4294967295,l=o==4294967295,u=t+r,f=a+c+l;if(n&&f){for(;t+4<u;t+=4+rt(e,t+2))if(rt(e,t)==1)return[a?vf(e,t+4+8*c):i,c?vf(e,t+4):s,l?vf(e,t+4+8*(c+a)):o,1];n<2&&K(13)}return[i,s,o,0]},Kr=function(e){var t=0;if(e)for(var r in e){var n=e[r].length;n>65535&&K(9),t+=n+4}return t},li=function(e,t,r,n,i,s,o,a){var c=n.length,l=r.extra,u=a&&a.length,f=Kr(l);he(e,t,o!=null?33639248:67324752),t+=4,o!=null&&(e[t++]=20,e[t++]=r.os),e[t]=20,t+=2,e[t++]=r.flag<<1|(s<0&&8),e[t++]=i&&8,e[t++]=r.compression&255,e[t++]=r.compression>>8;var h=new Date(r.mtime==null?Date.now():r.mtime),p=h.getFullYear()-1980;if((p<0||p>119)&&K(10),he(e,t,p<<25|h.getMonth()+1<<21|h.getDate()<<16|h.getHours()<<11|h.getMinutes()<<5|h.getSeconds()>>1),t+=4,s!=-1&&(he(e,t,r.crc),he(e,t+4,s<0?-s-2:s),he(e,t+8,r.size)),he(e,t+12,c),he(e,t+14,f),t+=16,o!=null&&(he(e,t,u),he(e,t+6,r.attrs),he(e,t+10,o),t+=14),e.set(n,t),t+=c,f)for(var d in l){var m=l[d],g=m.length;he(e,t,+d),he(e,t+2,g),e.set(m,t+4),t+=4+g}return u&&(e.set(a,t),t+=u),t},Lf=function(e,t,r,n,i){he(e,t,101010256),he(e,t+8,r),he(e,t+10,r),he(e,t+12,n),he(e,t+16,i)},ds=function(){function e(t){this.filename=t,this.c=pi(),this.size=0,this.compression=0}return e.prototype.process=function(t,r){this.ondata(null,t,r)},e.prototype.push=function(t,r){this.ondata||K(5),this.c.p(t),this.size+=t.length,r&&(this.crc=this.c.d()),this.process(t,r||!1)},e}();H.ZipPassThrough=ds;var Mk=function(){function e(t,r){var n=this;r||(r={}),ds.call(this,t),this.d=new Ut(r,function(i,s){n.ondata(null,i,s)}),this.compression=8,this.flag=_x(r.level)}return e.prototype.process=function(t,r){try{this.d.push(t,r)}catch(n){this.ondata(n,null,r)}},e.prototype.push=function(t,r){ds.prototype.push.call(this,t,r)},e}();H.ZipDeflate=Mk;var kk=function(){function e(t,r){var n=this;r||(r={}),ds.call(this,t),this.d=new gx(r,function(i,s,o){n.ondata(i,s,o)}),this.compression=8,this.flag=_x(r.level),this.terminate=this.d.terminate}return e.prototype.process=function(t,r){this.d.push(t,r)},e.prototype.push=function(t,r){ds.prototype.push.call(this,t,r)},e}();H.AsyncZipDeflate=kk;var Tk=function(){function e(t){this.ondata=t,this.u=[],this.d=1}return e.prototype.add=function(t){var r=this;if(this.ondata||K(5),this.d&2)this.ondata(K(4+(this.d&1)*8,0,1),null,!1);else{var n=Yr(t.filename),i=n.length,s=t.comment,o=s&&Yr(s),a=i!=t.filename.length||o&&s.length!=o.length,c=i+Kr(t.extra)+30;i>65535&&this.ondata(K(11,0,1),null,!1);var l=new ee(c);li(l,0,t,n,a,-1);var u=[l],f=function(){for(var g=0,y=u;g<y.length;g++){var I=y[g];r.ondata(null,I,!1)}u=[]},h=this.d;this.d=0;var p=this.u.length,d=ys(t,{f:n,u:a,o,t:function(){t.terminate&&t.terminate()},r:function(){if(f(),h){var g=r.u[p+1];g?g.r():r.d=1}h=1}}),m=0;t.ondata=function(g,y,I){if(g)r.ondata(g,y,I),r.terminate();else if(m+=y.length,u.push(y),I){var _=new ee(16);he(_,0,134695760),he(_,4,t.crc),he(_,8,m),he(_,12,t.size),u.push(_),d.c=m,d.b=c+m+16,d.crc=t.crc,d.size=t.size,h&&d.r(),h=1}else h&&f()},this.u.push(d)}},e.prototype.end=function(){var t=this;if(this.d&2){this.ondata(K(4+(this.d&1)*8,0,1),null,!0);return}this.d?this.e():this.u.push({r:function(){t.d&1&&(t.u.splice(-1,1),t.e())},t:function(){}}),this.d=3},e.prototype.e=function(){for(var t=0,r=0,n=0,i=0,s=this.u;i<s.length;i++){var o=s[i];n+=46+o.f.length+Kr(o.extra)+(o.o?o.o.length:0)}for(var a=new ee(n+22),c=0,l=this.u;c<l.length;c++){var o=l[c];li(a,t,o,o.f,o.u,-o.c-2,r,o.o),t+=46+o.f.length+Kr(o.extra)+(o.o?o.o.length:0),r+=o.b}Lf(a,t,this.u.length,n,r),this.ondata(null,a,!0),this.d=2},e.prototype.terminate=function(){for(var t=0,r=this.u;t<r.length;t++){var n=r[t];n.t()}this.d=2},e}();H.Zip=Tk;function Nk(e,t,r){r||(r=t,t={}),typeof r!="function"&&K(7);var n={};Of(e,"",n,t);var i=Object.keys(n),s=i.length,o=0,a=0,c=s,l=new Array(s),u=[],f=function(){for(var g=0;g<u.length;++g)u[g]()},h=function(g,y){Va(function(){r(g,y)})};Va(function(){h=r});var p=function(){var g=new ee(a+22),y=o,I=a-o;a=0;for(var _=0;_<c;++_){var E=l[_];try{var b=E.c.length;li(g,a,E,E.f,E.u,b);var C=30+E.f.length+Kr(E.extra),v=a+C;g.set(E.c,v),li(g,o,E,E.f,E.u,b,a,E.m),o+=16+C+(E.m?E.m.length:0),a=v+b}catch(w){return h(w,null)}}Lf(g,o,l.length,I,y),h(null,g)};s||p();for(var d=function(g){var y=i[g],I=n[y],_=I[0],E=I[1],b=pi(),C=_.length;b.p(_);var v=Yr(y),w=v.length,x=E.comment,k=x&&Yr(x),M=k&&k.length,P=Kr(E.extra),F=E.level==0?0:8,S=function(O,R){if(O)f(),h(O,null);else{var N=R.length;l[g]=ys(E,{size:C,crc:b.d(),c:R,f:v,m:k,u:w!=y.length||k&&x.length!=M,compression:F}),o+=30+w+P+N,a+=76+2*(w+P)+(M||0)+N,--s||p()}};if(w>65535&&S(K(11,0,1),null),!F)S(null,_);else if(C<16e4)try{S(null,vs(_,E))}catch(O){S(O,null)}else u.push(yx(_,E,S))},m=0;m<c;++m)d(m);return f}function Dk(e,t){t||(t={});var r={},n=[];Of(e,"",r,t);var i=0,s=0;for(var o in r){var a=r[o],c=a[0],l=a[1],u=l.level==0?0:8,f=Yr(o),h=f.length,p=l.comment,d=p&&Yr(p),m=d&&d.length,g=Kr(l.extra);h>65535&&K(11);var y=u?vs(c,l):c,I=y.length,_=pi();_.p(c),n.push(ys(l,{size:c.length,crc:_.d(),c:y,f,m:d,u:h!=o.length||d&&p.length!=m,o:i,compression:u})),i+=30+h+g+I,s+=76+2*(h+g)+(m||0)+I}for(var E=new ee(s+22),b=i,C=s-i,v=0;v<n.length;++v){var f=n[v];li(E,f.o,f,f.f,f.u,f.c.length);var w=30+f.f.length+Kr(f.extra);E.set(f.c,f.o+w),li(E,i,f,f.f,f.u,f.c.length,f.o,f.m),i+=16+w+(f.m?f.m.length:0)}return Lf(E,i,n.length,C,b),E}var kx=function(){function e(){}return e.prototype.push=function(t,r){this.ondata(null,t,r)},e.compression=0,e}();H.UnzipPassThrough=kx;var Pk=function(){function e(){var t=this;this.i=new it(function(r,n){t.ondata(null,r,n)})}return e.prototype.push=function(t,r){try{this.i.push(t,r)}catch(n){this.ondata(n,null,r)}},e.compression=8,e}();H.UnzipInflate=Pk;var Rk=function(){function e(t,r){var n=this;r<32e4?this.i=new it(function(i,s){n.ondata(null,i,s)}):(this.i=new Pf(function(i,s,o){n.ondata(i,s,o)}),this.terminate=this.i.terminate)}return e.prototype.push=function(t,r){this.i.terminate&&(t=Ot(t,0)),this.i.push(t,r)},e.compression=8,e}();H.AsyncUnzipInflate=Rk;var Ok=function(){function e(t){this.onfile=t,this.k=[],this.o={0:kx},this.p=cr}return e.prototype.push=function(t,r){var n=this;if(this.onfile||K(5),this.p||K(4),this.c>0){var i=Math.min(this.c,t.length),s=t.subarray(0,i);if(this.c-=i,this.d?this.d.push(s,!this.c):this.k[0].push(s),t=t.subarray(i),t.length)return this.push(t,r)}else{var o=0,a=0,c=void 0,l=void 0;this.p.length?t.length?(l=new ee(this.p.length+t.length),l.set(this.p),l.set(t,this.p.length)):l=this.p:l=t;for(var u=l.length,f=this.c,h=f&&this.d,p=function(){var y=Ie(l,a);if(y==67324752){o=1,c=a,d.d=null,d.c=0;var I=rt(l,a+6),_=rt(l,a+8),E=I&2048,b=I&8,C=rt(l,a+26),v=rt(l,a+28);if(u>a+30+C+v){var w=[];d.k.unshift(w),o=2;var x=Ie(l,a+18),k=Ie(l,a+22),M=Uf(l.subarray(a+30,a+=30+C),!E),P=Mx(l,a,v,2,x,k,0),F=P[0],S=P[1],O=P[3];b&&(F=-1-O),a+=v,d.c=F;var R,N={name:M,compression:_,start:function(){if(N.ondata||K(5),!F)N.ondata(null,cr,!0);else{var U=n.o[_];U||N.ondata(K(14,"unknown compression type "+_,1),null,!1),R=F<0?new U(M):new U(M,F,S),R.ondata=function(Y,J,de){N.ondata(Y,J,de)};for(var L=0,G=w;L<G.length;L++){var V=G[L];R.push(V,!1)}n.k[0]==w&&n.c?n.d=R:R.push(cr,!0)}},terminate:function(){R&&R.terminate&&R.terminate()}};F>=0&&(N.size=F,N.originalSize=S),d.onfile(N)}return"break"}else if(f){if(y==134695760)return c=a+=12+(f==-2&&8),o=3,d.c=0,"break";if(y==33639248)return c=a-=4,o=3,d.c=0,"break"}},d=this;a<u-4;++a){var m=p();if(m==="break")break}if(this.p=cr,f<0){var g=o?l.subarray(0,c-12-(f==-2&&8)-(Ie(l,c-16)==134695760&&4)):l.subarray(0,a);h?h.push(g,!!o):this.k[+(o==2)].push(g)}if(o&2)return this.push(l.subarray(a),r);this.p=l.subarray(a)}r&&(this.c&&K(13),this.p=null)},e.prototype.register=function(t){this.o[t.compression]=t},e}();H.Unzip=Ok;var Va=typeof queueMicrotask=="function"?queueMicrotask:typeof setTimeout=="function"?setTimeout:function(e){e()};function Uk(e,t,r){r||(r=t,t={}),typeof r!="function"&&K(7);var n=[],i=function(){for(var g=0;g<n.length;++g)n[g]()},s={},o=function(g,y){Va(function(){r(g,y)})};Va(function(){o=r});for(var a=e.length-22;Ie(e,a)!=101010256;--a)if(!a||e.length-a>65558)return o(K(13,0,1),null),i;var c=rt(e,a+8);if(c){var l=c,u=Ie(e,a+16),f=Ie(e,a-20)==117853008;if(f){var h=Ie(e,a-12);f=Ie(e,h)==101075792,f&&(l=c=Ie(e,h+32),u=Ie(e,h+48))}for(var p=t&&t.filter,d=function(g){var y=Ax(e,u,f),I=y[0],_=y[1],E=y[2],b=y[3],C=y[4],v=y[5],w=Cx(e,v);u=C;var x=function(M,P){M?(i(),o(M,null)):(P&&(s[b]=P),--c||o(null,s))};if(!p||p({name:b,size:_,originalSize:E,compression:I}))if(!I)x(null,Ot(e,w,w+_));else if(I==8){var k=e.subarray(w,w+_);if(E<524288||_>.8*E)try{x(null,vi(k,{out:new ee(E)}))}catch(M){x(M,null)}else n.push(Rf(k,{size:E},x))}else x(K(14,"unknown compression type "+I,1),null);else x(null,null)},m=0;m<l;++m)d(m)}else o(null,{});return i}function Lk(e,t){for(var r={},n=e.length-22;Ie(e,n)!=101010256;--n)(!n||e.length-n>65558)&&K(13);var i=rt(e,n+8);if(!i)return{};var s=Ie(e,n+16),o=Ie(e,n-20)==117853008;if(o){var a=Ie(e,n-12);o=Ie(e,a)==101075792,o&&(i=Ie(e,a+32),s=Ie(e,a+48))}for(var c=t&&t.filter,l=0;l<i;++l){var u=Ax(e,s,o),f=u[0],h=u[1],p=u[2],d=u[3],m=u[4],g=u[5],y=Cx(e,g);s=m,(!c||c({name:d,size:h,originalSize:p,compression:f}))&&(f?f==8?r[d]=vi(e.subarray(y,y+h),{out:new ee(p)}):K(14,"unknown compression type "+f):r[d]=Ot(e,y,y+h))}return r}});var Bf=q(Ka=>{"use strict";Object.defineProperty(Ka,"__esModule",{value:!0});Ka.NIFTIEXTENSION=void 0;var Ff=class{esize;ecode;edata;littleEndian;constructor(t,r,n,i){if(t%16!=0)throw new Error("This does not appear to be a NIFTI extension");this.esize=t,this.ecode=r,this.edata=n,this.littleEndian=i}toArrayBuffer(){let t=new Uint8Array(this.esize),r=new Uint8Array(this.edata);t.set(r,8);let n=new DataView(t.buffer);return n.setInt32(0,this.esize,this.littleEndian),n.setInt32(4,this.ecode,this.littleEndian),t.buffer}};Ka.NIFTIEXTENSION=Ff});var xs=q(Ya=>{"use strict";Object.defineProperty(Ya,"__esModule",{value:!0});Ya.Utils=void 0;var Fk=Bf(),zf=class e{static crcTable=null;static GUNZIP_MAGIC_COOKIE1=31;static GUNZIP_MAGIC_COOKIE2=139;static getStringAt(t,r,n){var i="",s,o;for(s=r;s<n;s+=1)o=t.getUint8(s),o!==0&&(i+=String.fromCharCode(o));return i}static getByteAt=function(t,r){return t.getUint8(r)};static getShortAt=function(t,r,n){return t.getInt16(r,n)};static getIntAt(t,r,n){return t.getInt32(r,n)}static getFloatAt(t,r,n){return t.getFloat32(r,n)}static getDoubleAt(t,r,n){return t.getFloat64(r,n)}static getInt64At(t,r,n){let i=t.getUint32(r,n),s=t.getInt32(r+4,n),o;return n?o=s*2**32+i:o=i*2**32+s,s<0&&(o+=-1*2**32*2**32),o}static getExtensionsAt(t,r,n,i){let s=[],o=r;for(;o<i;){let a=n,c=e.getIntAt(t,o,n);if(!c)break;if(c+o>i&&(a=!a,c=e.getIntAt(t,o,a),c+o>i))throw new Error("This does not appear to be a valid NIFTI extension");if(c%16!=0)throw new Error("This does not appear to be a NIFTI extension");let l=e.getIntAt(t,o+4,a),u=t.buffer.slice(o+8,o+c);console.log("extensionByteIndex: "+(o+8)+" esize: "+c),console.log(u);let f=new Fk.NIFTIEXTENSION(c,l,u,a);s.push(f),o+=c}return s}static toArrayBuffer(t){var r,n,i;for(r=new ArrayBuffer(t.length),n=new Uint8Array(r),i=0;i<t.length;i+=1)n[i]=t[i];return r}static isString(t){return typeof t=="string"||t instanceof String}static formatNumber(t,r=void 0){let n;return e.isString(t)?n=Number(t):n=t,r?n=n.toPrecision(5):n=n.toPrecision(7),parseFloat(n)}static makeCRCTable(){let t,r=[];for(var n=0;n<256;n++){t=n;for(var i=0;i<8;i++)t=t&1?3988292384^t>>>1:t>>>1;r[n]=t}return r}static crc32(t){e.crcTable||(e.crcTable=e.makeCRCTable());let r=e.crcTable,n=-1;for(var i=0;i<t.byteLength;i++)n=n>>>8^r[(n^t.getUint8(i))&255];return(n^-1)>>>0}};Ya.Utils=zf});var Ha=q(Ja=>{"use strict";Object.defineProperty(Ja,"__esModule",{value:!0});Ja.NIFTI1=void 0;var Q=xs(),$f=class e{littleEndian=!1;dim_info=0;dims=[];intent_p1=0;intent_p2=0;intent_p3=0;intent_code=0;datatypeCode=0;numBitsPerVoxel=0;slice_start=0;slice_end=0;slice_code=0;pixDims=[];vox_offset=0;scl_slope=1;scl_inter=0;xyzt_units=0;cal_max=0;cal_min=0;slice_duration=0;toffset=0;description="";aux_file="";intent_name="";qform_code=0;sform_code=0;quatern_a=0;quatern_b=0;quatern_c=0;quatern_d=0;qoffset_x=0;qoffset_y=0;qoffset_z=0;affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];qfac=1;quatern_R;magic="0";isHDR=!1;extensionFlag=[0,0,0,0];extensionSize=0;extensionCode=0;extensions=[];static TYPE_NONE=0;static TYPE_BINARY=1;static TYPE_UINT8=2;static TYPE_INT16=4;static TYPE_INT32=8;static TYPE_FLOAT32=16;static TYPE_COMPLEX64=32;static TYPE_FLOAT64=64;static TYPE_RGB24=128;static TYPE_INT8=256;static TYPE_UINT16=512;static TYPE_UINT32=768;static TYPE_INT64=1024;static TYPE_UINT64=1280;static TYPE_FLOAT128=1536;static TYPE_COMPLEX128=1792;static TYPE_COMPLEX256=2048;static XFORM_UNKNOWN=0;static XFORM_SCANNER_ANAT=1;static XFORM_ALIGNED_ANAT=2;static XFORM_TALAIRACH=3;static XFORM_MNI_152=4;static SPATIAL_UNITS_MASK=7;static TEMPORAL_UNITS_MASK=56;static UNITS_UNKNOWN=0;static UNITS_METER=1;static UNITS_MM=2;static UNITS_MICRON=3;static UNITS_SEC=8;static UNITS_MSEC=16;static UNITS_USEC=24;static UNITS_HZ=32;static UNITS_PPM=40;static UNITS_RADS=48;static MAGIC_COOKIE=348;static STANDARD_HEADER_SIZE=348;static MAGIC_NUMBER_LOCATION=344;static MAGIC_NUMBER=[110,43,49];static MAGIC_NUMBER2=[110,105,49];static EXTENSION_HEADER_SIZE=8;readHeader(t){var r=new DataView(t),n=Q.Utils.getIntAt(r,0,this.littleEndian),i,s,o,a;if(n!==e.MAGIC_COOKIE&&(this.littleEndian=!0,n=Q.Utils.getIntAt(r,0,this.littleEndian)),n!==e.MAGIC_COOKIE)throw new Error("This does not appear to be a NIFTI file!");for(this.dim_info=Q.Utils.getByteAt(r,39),i=0;i<8;i+=1)a=40+i*2,this.dims[i]=Q.Utils.getShortAt(r,a,this.littleEndian);for(this.intent_p1=Q.Utils.getFloatAt(r,56,this.littleEndian),this.intent_p2=Q.Utils.getFloatAt(r,60,this.littleEndian),this.intent_p3=Q.Utils.getFloatAt(r,64,this.littleEndian),this.intent_code=Q.Utils.getShortAt(r,68,this.littleEndian),this.datatypeCode=Q.Utils.getShortAt(r,70,this.littleEndian),this.numBitsPerVoxel=Q.Utils.getShortAt(r,72,this.littleEndian),this.slice_start=Q.Utils.getShortAt(r,74,this.littleEndian),i=0;i<8;i+=1)a=76+i*4,this.pixDims[i]=Q.Utils.getFloatAt(r,a,this.littleEndian);if(this.vox_offset=Q.Utils.getFloatAt(r,108,this.littleEndian),this.scl_slope=Q.Utils.getFloatAt(r,112,this.littleEndian),this.scl_inter=Q.Utils.getFloatAt(r,116,this.littleEndian),this.slice_end=Q.Utils.getShortAt(r,120,this.littleEndian),this.slice_code=Q.Utils.getByteAt(r,122),this.xyzt_units=Q.Utils.getByteAt(r,123),this.cal_max=Q.Utils.getFloatAt(r,124,this.littleEndian),this.cal_min=Q.Utils.getFloatAt(r,128,this.littleEndian),this.slice_duration=Q.Utils.getFloatAt(r,132,this.littleEndian),this.toffset=Q.Utils.getFloatAt(r,136,this.littleEndian),this.description=Q.Utils.getStringAt(r,148,228),this.aux_file=Q.Utils.getStringAt(r,228,252),this.qform_code=Q.Utils.getShortAt(r,252,this.littleEndian),this.sform_code=Q.Utils.getShortAt(r,254,this.littleEndian),this.quatern_b=Q.Utils.getFloatAt(r,256,this.littleEndian),this.quatern_c=Q.Utils.getFloatAt(r,260,this.littleEndian),this.quatern_d=Q.Utils.getFloatAt(r,264,this.littleEndian),this.quatern_a=Math.sqrt(1-(Math.pow(this.quatern_b,2)+Math.pow(this.quatern_c,2)+Math.pow(this.quatern_d,2))),this.qoffset_x=Q.Utils.getFloatAt(r,268,this.littleEndian),this.qoffset_y=Q.Utils.getFloatAt(r,272,this.littleEndian),this.qoffset_z=Q.Utils.getFloatAt(r,276,this.littleEndian),this.qform_code<1&&this.sform_code<1&&(this.affine[0][0]=this.pixDims[1],this.affine[1][1]=this.pixDims[2],this.affine[2][2]=this.pixDims[3]),this.qform_code>0&&this.sform_code<this.qform_code){let c=this.quatern_a,l=this.quatern_b,u=this.quatern_c,f=this.quatern_d;for(this.qfac=this.pixDims[0]===0?1:this.pixDims[0],this.quatern_R=[[c*c+l*l-u*u-f*f,2*l*u-2*c*f,2*l*f+2*c*u],[2*l*u+2*c*f,c*c+u*u-l*l-f*f,2*u*f-2*c*l],[2*l*f-2*c*u,2*u*f+2*c*l,c*c+f*f-u*u-l*l]],s=0;s<3;s+=1)for(o=0;o<3;o+=1)this.affine[s][o]=this.quatern_R[s][o]*this.pixDims[o+1],o===2&&(this.affine[s][o]*=this.qfac);this.affine[0][3]=this.qoffset_x,this.affine[1][3]=this.qoffset_y,this.affine[2][3]=this.qoffset_z}else if(this.sform_code>0)for(s=0;s<3;s+=1)for(o=0;o<4;o+=1)a=280+(s*4+o)*4,this.affine[s][o]=Q.Utils.getFloatAt(r,a,this.littleEndian);if(this.affine[3][0]=0,this.affine[3][1]=0,this.affine[3][2]=0,this.affine[3][3]=1,this.intent_name=Q.Utils.getStringAt(r,328,344),this.magic=Q.Utils.getStringAt(r,344,348),this.isHDR=this.magic===String.fromCharCode.apply(null,e.MAGIC_NUMBER2),r.byteLength>e.MAGIC_COOKIE){this.extensionFlag[0]=Q.Utils.getByteAt(r,348),this.extensionFlag[1]=Q.Utils.getByteAt(r,349),this.extensionFlag[2]=Q.Utils.getByteAt(r,350),this.extensionFlag[3]=Q.Utils.getByteAt(r,351);let c=!0;!this.isHDR&&this.vox_offset<=352&&(c=!1),r.byteLength<=368&&(c=!1),c&&this.extensionFlag[0]&&(this.extensions=Q.Utils.getExtensionsAt(r,this.getExtensionLocation(),this.littleEndian,this.vox_offset),this.extensionSize=this.extensions[0].esize,this.extensionCode=this.extensions[0].ecode)}}toFormattedString(){var t=Q.Utils.formatNumber,r="";return r+="Dim Info = "+this.dim_info+`
`,r+="Image Dimensions (1-8): "+this.dims[0]+", "+this.dims[1]+", "+this.dims[2]+", "+this.dims[3]+", "+this.dims[4]+", "+this.dims[5]+", "+this.dims[6]+", "+this.dims[7]+`
`,r+="Intent Parameters (1-3): "+this.intent_p1+", "+this.intent_p2+", "+this.intent_p3+`
`,r+="Intent Code = "+this.intent_code+`
`,r+="Datatype = "+this.datatypeCode+" ("+this.getDatatypeCodeString(this.datatypeCode)+`)
`,r+="Bits Per Voxel = "+this.numBitsPerVoxel+`
`,r+="Slice Start = "+this.slice_start+`
`,r+="Voxel Dimensions (1-8): "+t(this.pixDims[0])+", "+t(this.pixDims[1])+", "+t(this.pixDims[2])+", "+t(this.pixDims[3])+", "+t(this.pixDims[4])+", "+t(this.pixDims[5])+", "+t(this.pixDims[6])+", "+t(this.pixDims[7])+`
`,r+="Image Offset = "+this.vox_offset+`
`,r+="Data Scale:  Slope = "+t(this.scl_slope)+"  Intercept = "+t(this.scl_inter)+`
`,r+="Slice End = "+this.slice_end+`
`,r+="Slice Code = "+this.slice_code+`
`,r+="Units Code = "+this.xyzt_units+" ("+this.getUnitsCodeString(e.SPATIAL_UNITS_MASK&this.xyzt_units)+", "+this.getUnitsCodeString(e.TEMPORAL_UNITS_MASK&this.xyzt_units)+`)
`,r+="Display Range:  Max = "+t(this.cal_max)+"  Min = "+t(this.cal_min)+`
`,r+="Slice Duration = "+this.slice_duration+`
`,r+="Time Axis Shift = "+this.toffset+`
`,r+='Description: "'+this.description+`"
`,r+='Auxiliary File: "'+this.aux_file+`"
`,r+="Q-Form Code = "+this.qform_code+" ("+this.getTransformCodeString(this.qform_code)+`)
`,r+="S-Form Code = "+this.sform_code+" ("+this.getTransformCodeString(this.sform_code)+`)
`,r+="Quaternion Parameters:  b = "+t(this.quatern_b)+"  c = "+t(this.quatern_c)+"  d = "+t(this.quatern_d)+`
`,r+="Quaternion Offsets:  x = "+this.qoffset_x+"  y = "+this.qoffset_y+"  z = "+this.qoffset_z+`
`,r+="S-Form Parameters X: "+t(this.affine[0][0])+", "+t(this.affine[0][1])+", "+t(this.affine[0][2])+", "+t(this.affine[0][3])+`
`,r+="S-Form Parameters Y: "+t(this.affine[1][0])+", "+t(this.affine[1][1])+", "+t(this.affine[1][2])+", "+t(this.affine[1][3])+`
`,r+="S-Form Parameters Z: "+t(this.affine[2][0])+", "+t(this.affine[2][1])+", "+t(this.affine[2][2])+", "+t(this.affine[2][3])+`
`,r+='Intent Name: "'+this.intent_name+`"
`,this.extensionFlag[0]&&(r+="Extension: Size = "+this.extensionSize+"  Code = "+this.extensionCode+`
`),r}getDatatypeCodeString=function(t){return t===e.TYPE_UINT8?"1-Byte Unsigned Integer":t===e.TYPE_INT16?"2-Byte Signed Integer":t===e.TYPE_INT32?"4-Byte Signed Integer":t===e.TYPE_FLOAT32?"4-Byte Float":t===e.TYPE_FLOAT64?"8-Byte Float":t===e.TYPE_RGB24?"RGB":t===e.TYPE_INT8?"1-Byte Signed Integer":t===e.TYPE_UINT16?"2-Byte Unsigned Integer":t===e.TYPE_UINT32?"4-Byte Unsigned Integer":t===e.TYPE_INT64?"8-Byte Signed Integer":t===e.TYPE_UINT64?"8-Byte Unsigned Integer":"Unknown"};getTransformCodeString=function(t){return t===e.XFORM_SCANNER_ANAT?"Scanner":t===e.XFORM_ALIGNED_ANAT?"Aligned":t===e.XFORM_TALAIRACH?"Talairach":t===e.XFORM_MNI_152?"MNI":"Unknown"};getUnitsCodeString=function(t){return t===e.UNITS_METER?"Meters":t===e.UNITS_MM?"Millimeters":t===e.UNITS_MICRON?"Microns":t===e.UNITS_SEC?"Seconds":t===e.UNITS_MSEC?"Milliseconds":t===e.UNITS_USEC?"Microseconds":t===e.UNITS_HZ?"Hz":t===e.UNITS_PPM?"PPM":t===e.UNITS_RADS?"Rads":"Unknown"};getQformMat(){return this.convertNiftiQFormToNiftiSForm(this.quatern_b,this.quatern_c,this.quatern_d,this.qoffset_x,this.qoffset_y,this.qoffset_z,this.pixDims[1],this.pixDims[2],this.pixDims[3],this.pixDims[0])}convertNiftiQFormToNiftiSForm(t,r,n,i,s,o,a,c,l,u){var f=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],h,p=t,d=r,m=n,g,y,I;return f[3][0]=f[3][1]=f[3][2]=0,f[3][3]=1,h=1-(p*p+d*d+m*m),h<1e-7?(h=1/Math.sqrt(p*p+d*d+m*m),p*=h,d*=h,m*=h,h=0):h=Math.sqrt(h),g=a>0?a:1,y=c>0?c:1,I=l>0?l:1,u<0&&(I=-I),f[0][0]=(h*h+p*p-d*d-m*m)*g,f[0][1]=2*(p*d-h*m)*y,f[0][2]=2*(p*m+h*d)*I,f[1][0]=2*(p*d+h*m)*g,f[1][1]=(h*h+d*d-p*p-m*m)*y,f[1][2]=2*(d*m-h*p)*I,f[2][0]=2*(p*m-h*d)*g,f[2][1]=2*(d*m+h*p)*y,f[2][2]=(h*h+m*m-d*d-p*p)*I,f[0][3]=i,f[1][3]=s,f[2][3]=o,f}convertNiftiSFormToNEMA(t){var r,n,i,s,o,a,c,l,u,f,h,p,d,m,g,y,I,_,E,b,C,v,w,x,k,M,P,F,S,O,R,N,U,L;if(g=0,P=[[0,0,0],[0,0,0],[0,0,0]],F=[[0,0,0],[0,0,0],[0,0,0]],r=t[0][0],n=t[0][1],i=t[0][2],s=t[1][0],o=t[1][1],a=t[1][2],c=t[2][0],l=t[2][1],u=t[2][2],f=Math.sqrt(r*r+s*s+c*c),f===0||(r/=f,s/=f,c/=f,f=Math.sqrt(n*n+o*o+l*l),f===0))return null;if(n/=f,o/=f,l/=f,f=r*n+s*o+c*l,Math.abs(f)>1e-4){if(n-=f*r,o-=f*s,l-=f*c,f=Math.sqrt(n*n+o*o+l*l),f===0)return null;n/=f,o/=f,l/=f}if(f=Math.sqrt(i*i+a*a+u*u),f===0?(i=s*l-c*o,a=c*n-l*r,u=r*o-s*n):(i/=f,a/=f,u/=f),f=r*i+s*a+c*u,Math.abs(f)>1e-4){if(i-=f*r,a-=f*s,u-=f*c,f=Math.sqrt(i*i+a*a+u*u),f===0)return null;i/=f,a/=f,u/=f}if(f=n*i+o*a+l*u,Math.abs(f)>1e-4){if(i-=f*n,a-=f*o,u-=f*l,f=Math.sqrt(i*i+a*a+u*u),f===0)return null;i/=f,a/=f,u/=f}if(P[0][0]=r,P[0][1]=n,P[0][2]=i,P[1][0]=s,P[1][1]=o,P[1][2]=a,P[2][0]=c,P[2][1]=l,P[2][2]=u,h=this.nifti_mat33_determ(P),h===0)return null;for(M=-666,E=v=w=x=1,b=2,C=3,d=1;d<=3;d+=1)for(m=1;m<=3;m+=1)if(d!==m){for(g=1;g<=3;g+=1)if(!(d===g||m===g))for(F[0][0]=F[0][1]=F[0][2]=F[1][0]=F[1][1]=F[1][2]=F[2][0]=F[2][1]=F[2][2]=0,y=-1;y<=1;y+=2)for(I=-1;I<=1;I+=2)for(_=-1;_<=1;_+=2)F[0][d-1]=y,F[1][m-1]=I,F[2][g-1]=_,p=this.nifti_mat33_determ(F),p*h>0&&(k=this.nifti_mat33_mul(F,P),f=k[0][0]+k[1][1]+k[2][2],f>M&&(M=f,E=d,b=m,C=g,v=y,w=I,x=_))}switch(S=O=R=N=U=L="",E*v){case 1:S="X",N="+";break;case-1:S="X",N="-";break;case 2:S="Y",N="+";break;case-2:S="Y",N="-";break;case 3:S="Z",N="+";break;case-3:S="Z",N="-";break}switch(b*w){case 1:O="X",U="+";break;case-1:O="X",U="-";break;case 2:O="Y",U="+";break;case-2:O="Y",U="-";break;case 3:O="Z",U="+";break;case-3:O="Z",U="-";break}switch(C*x){case 1:R="X",L="+";break;case-1:R="X",L="-";break;case 2:R="Y",L="+";break;case-2:R="Y",L="-";break;case 3:R="Z",L="+";break;case-3:R="Z",L="-";break}return S+O+R+N+U+L}nifti_mat33_mul=function(t,r){var n=[[0,0,0],[0,0,0],[0,0,0]],i,s;for(i=0;i<3;i+=1)for(s=0;s<3;s+=1)n[i][s]=t[i][0]*r[0][s]+t[i][1]*r[1][s]+t[i][2]*r[2][s];return n};nifti_mat33_determ=function(t){var r,n,i,s,o,a,c,l,u;return r=t[0][0],n=t[0][1],i=t[0][2],s=t[1][0],o=t[1][1],a=t[1][2],c=t[2][0],l=t[2][1],u=t[2][2],r*o*u-r*l*a-s*n*u+s*l*i+c*n*a-c*o*i};getExtensionLocation(){return e.MAGIC_COOKIE+4}getExtensionSize(t){return Q.Utils.getIntAt(t,this.getExtensionLocation(),this.littleEndian)}getExtensionCode(t){return Q.Utils.getIntAt(t,this.getExtensionLocation()+4,this.littleEndian)}addExtension(t,r=-1){r==-1?this.extensions.push(t):this.extensions.splice(r,0,t),this.vox_offset+=t.esize}removeExtension(t){let r=this.extensions[t];r&&(this.vox_offset-=r.esize),this.extensions.splice(t,1)}toArrayBuffer(t=!1){let i=352;if(t)for(let c of this.extensions)i+=c.esize;let s=new Uint8Array(i),o=new DataView(s.buffer);o.setInt32(0,348,this.littleEndian),o.setUint8(39,this.dim_info);for(let c=0;c<8;c++)o.setUint16(40+2*c,this.dims[c],this.littleEndian);o.setFloat32(56,this.intent_p1,this.littleEndian),o.setFloat32(60,this.intent_p2,this.littleEndian),o.setFloat32(64,this.intent_p3,this.littleEndian),o.setInt16(68,this.intent_code,this.littleEndian),o.setInt16(70,this.datatypeCode,this.littleEndian),o.setInt16(72,this.numBitsPerVoxel,this.littleEndian),o.setInt16(74,this.slice_start,this.littleEndian);for(let c=0;c<8;c++)o.setFloat32(76+4*c,this.pixDims[c],this.littleEndian);o.setFloat32(108,this.vox_offset,this.littleEndian),o.setFloat32(112,this.scl_slope,this.littleEndian),o.setFloat32(116,this.scl_inter,this.littleEndian),o.setInt16(120,this.slice_end,this.littleEndian),o.setUint8(122,this.slice_code),o.setUint8(123,this.xyzt_units),o.setFloat32(124,this.cal_max,this.littleEndian),o.setFloat32(128,this.cal_min,this.littleEndian),o.setFloat32(132,this.slice_duration,this.littleEndian),o.setFloat32(136,this.toffset,this.littleEndian),s.set(Buffer.from(this.description),148),s.set(Buffer.from(this.aux_file),228),o.setInt16(252,this.qform_code,this.littleEndian),o.setInt16(254,this.sform_code,this.littleEndian),o.setFloat32(256,this.quatern_b,this.littleEndian),o.setFloat32(260,this.quatern_c,this.littleEndian),o.setFloat32(264,this.quatern_d,this.littleEndian),o.setFloat32(268,this.qoffset_x,this.littleEndian),o.setFloat32(272,this.qoffset_y,this.littleEndian),o.setFloat32(276,this.qoffset_z,this.littleEndian);let a=this.affine.flat();for(let c=0;c<12;c++)o.setFloat32(280+4*c,a[c],this.littleEndian);if(s.set(Buffer.from(this.intent_name),328),s.set(Buffer.from(this.magic),344),t){s.set(Uint8Array.from([1,0,0,0]),348);let c=this.getExtensionLocation();for(let l of this.extensions)o.setInt32(c,l.esize,l.littleEndian),o.setInt32(c+4,l.ecode,l.littleEndian),s.set(new Uint8Array(l.edata),c+8),c+=l.esize}else s.set(new Uint8Array(4).fill(0),348);return s.buffer}};Ja.NIFTI1=$f});var jf=q(Wa=>{"use strict";Object.defineProperty(Wa,"__esModule",{value:!0});Wa.NIFTI2=void 0;var ht=Ha(),te=xs(),Vf=class e{littleEndian=!1;dim_info=0;dims=[];intent_p1=0;intent_p2=0;intent_p3=0;intent_code=0;datatypeCode=0;numBitsPerVoxel=0;slice_start=0;slice_end=0;slice_code=0;pixDims=[];vox_offset=0;scl_slope=1;scl_inter=0;xyzt_units=0;cal_max=0;cal_min=0;slice_duration=0;toffset=0;description="";aux_file="";intent_name="";qform_code=0;sform_code=0;quatern_b=0;quatern_c=0;quatern_d=0;qoffset_x=0;qoffset_y=0;qoffset_z=0;affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];magic="0";extensionFlag=[0,0,0,0];extensions=[];extensionSize=0;extensionCode=0;static MAGIC_COOKIE=540;static MAGIC_NUMBER_LOCATION=4;static MAGIC_NUMBER=[110,43,50,0,13,10,26,10];static MAGIC_NUMBER2=[110,105,50,0,13,10,26,10];readHeader(t){var r=new DataView(t),n=te.Utils.getIntAt(r,0,this.littleEndian),i,s,o,a,c;if(n!==e.MAGIC_COOKIE&&(this.littleEndian=!0,n=te.Utils.getIntAt(r,0,this.littleEndian)),n!==e.MAGIC_COOKIE)throw new Error("This does not appear to be a NIFTI file!");for(this.magic=te.Utils.getStringAt(r,4,12),this.datatypeCode=te.Utils.getShortAt(r,12,this.littleEndian),this.numBitsPerVoxel=te.Utils.getShortAt(r,14,this.littleEndian),i=0;i<8;i+=1)a=16+i*8,this.dims[i]=te.Utils.getInt64At(r,a,this.littleEndian);for(this.intent_p1=te.Utils.getDoubleAt(r,80,this.littleEndian),this.intent_p2=te.Utils.getDoubleAt(r,88,this.littleEndian),this.intent_p3=te.Utils.getDoubleAt(r,96,this.littleEndian),i=0;i<8;i+=1)a=104+i*8,this.pixDims[i]=te.Utils.getDoubleAt(r,a,this.littleEndian);for(this.vox_offset=te.Utils.getInt64At(r,168,this.littleEndian),this.scl_slope=te.Utils.getDoubleAt(r,176,this.littleEndian),this.scl_inter=te.Utils.getDoubleAt(r,184,this.littleEndian),this.cal_max=te.Utils.getDoubleAt(r,192,this.littleEndian),this.cal_min=te.Utils.getDoubleAt(r,200,this.littleEndian),this.slice_duration=te.Utils.getDoubleAt(r,208,this.littleEndian),this.toffset=te.Utils.getDoubleAt(r,216,this.littleEndian),this.slice_start=te.Utils.getInt64At(r,224,this.littleEndian),this.slice_end=te.Utils.getInt64At(r,232,this.littleEndian),this.description=te.Utils.getStringAt(r,240,320),this.aux_file=te.Utils.getStringAt(r,320,344),this.qform_code=te.Utils.getIntAt(r,344,this.littleEndian),this.sform_code=te.Utils.getIntAt(r,348,this.littleEndian),this.quatern_b=te.Utils.getDoubleAt(r,352,this.littleEndian),this.quatern_c=te.Utils.getDoubleAt(r,360,this.littleEndian),this.quatern_d=te.Utils.getDoubleAt(r,368,this.littleEndian),this.qoffset_x=te.Utils.getDoubleAt(r,376,this.littleEndian),this.qoffset_y=te.Utils.getDoubleAt(r,384,this.littleEndian),this.qoffset_z=te.Utils.getDoubleAt(r,392,this.littleEndian),s=0;s<3;s+=1)for(o=0;o<4;o+=1)a=400+(s*4+o)*8,this.affine[s][o]=te.Utils.getDoubleAt(r,a,this.littleEndian);this.affine[3][0]=0,this.affine[3][1]=0,this.affine[3][2]=0,this.affine[3][3]=1,this.slice_code=te.Utils.getIntAt(r,496,this.littleEndian),this.xyzt_units=te.Utils.getIntAt(r,500,this.littleEndian),this.intent_code=te.Utils.getIntAt(r,504,this.littleEndian),this.intent_name=te.Utils.getStringAt(r,508,524),this.dim_info=te.Utils.getByteAt(r,524),r.byteLength>e.MAGIC_COOKIE&&(this.extensionFlag[0]=te.Utils.getByteAt(r,540),this.extensionFlag[1]=te.Utils.getByteAt(r,541),this.extensionFlag[2]=te.Utils.getByteAt(r,542),this.extensionFlag[3]=te.Utils.getByteAt(r,543),this.extensionFlag[0]&&(this.extensions=te.Utils.getExtensionsAt(r,this.getExtensionLocation(),this.littleEndian,this.vox_offset),this.extensionSize=this.extensions[0].esize,this.extensionCode=this.extensions[0].ecode))}toFormattedString(){var t=te.Utils.formatNumber,r="";return r+="Datatype = "+ +this.datatypeCode+" ("+this.getDatatypeCodeString(this.datatypeCode)+`)
`,r+="Bits Per Voxel =  = "+this.numBitsPerVoxel+`
`,r+="Image Dimensions (1-8): "+this.dims[0]+", "+this.dims[1]+", "+this.dims[2]+", "+this.dims[3]+", "+this.dims[4]+", "+this.dims[5]+", "+this.dims[6]+", "+this.dims[7]+`
`,r+="Intent Parameters (1-3): "+this.intent_p1+", "+this.intent_p2+", "+this.intent_p3+`
`,r+="Voxel Dimensions (1-8): "+t(this.pixDims[0])+", "+t(this.pixDims[1])+", "+t(this.pixDims[2])+", "+t(this.pixDims[3])+", "+t(this.pixDims[4])+", "+t(this.pixDims[5])+", "+t(this.pixDims[6])+", "+t(this.pixDims[7])+`
`,r+="Image Offset = "+this.vox_offset+`
`,r+="Data Scale:  Slope = "+t(this.scl_slope)+"  Intercept = "+t(this.scl_inter)+`
`,r+="Display Range:  Max = "+t(this.cal_max)+"  Min = "+t(this.cal_min)+`
`,r+="Slice Duration = "+this.slice_duration+`
`,r+="Time Axis Shift = "+this.toffset+`
`,r+="Slice Start = "+this.slice_start+`
`,r+="Slice End = "+this.slice_end+`
`,r+='Description: "'+this.description+`"
`,r+='Auxiliary File: "'+this.aux_file+`"
`,r+="Q-Form Code = "+this.qform_code+" ("+this.getTransformCodeString(this.qform_code)+`)
`,r+="S-Form Code = "+this.sform_code+" ("+this.getTransformCodeString(this.sform_code)+`)
`,r+="Quaternion Parameters:  b = "+t(this.quatern_b)+"  c = "+t(this.quatern_c)+"  d = "+t(this.quatern_d)+`
`,r+="Quaternion Offsets:  x = "+this.qoffset_x+"  y = "+this.qoffset_y+"  z = "+this.qoffset_z+`
`,r+="S-Form Parameters X: "+t(this.affine[0][0])+", "+t(this.affine[0][1])+", "+t(this.affine[0][2])+", "+t(this.affine[0][3])+`
`,r+="S-Form Parameters Y: "+t(this.affine[1][0])+", "+t(this.affine[1][1])+", "+t(this.affine[1][2])+", "+t(this.affine[1][3])+`
`,r+="S-Form Parameters Z: "+t(this.affine[2][0])+", "+t(this.affine[2][1])+", "+t(this.affine[2][2])+", "+t(this.affine[2][3])+`
`,r+="Slice Code = "+this.slice_code+`
`,r+="Units Code = "+this.xyzt_units+" ("+this.getUnitsCodeString(ht.NIFTI1.SPATIAL_UNITS_MASK&this.xyzt_units)+", "+this.getUnitsCodeString(ht.NIFTI1.TEMPORAL_UNITS_MASK&this.xyzt_units)+`)
`,r+="Intent Code = "+this.intent_code+`
`,r+='Intent Name: "'+this.intent_name+`"
`,r+="Dim Info = "+this.dim_info+`
`,r}getExtensionLocation=function(){return e.MAGIC_COOKIE+4};getExtensionSize=ht.NIFTI1.prototype.getExtensionSize;getExtensionCode=ht.NIFTI1.prototype.getExtensionCode;addExtension=ht.NIFTI1.prototype.addExtension;removeExtension=ht.NIFTI1.prototype.removeExtension;getDatatypeCodeString=ht.NIFTI1.prototype.getDatatypeCodeString;getTransformCodeString=ht.NIFTI1.prototype.getTransformCodeString;getUnitsCodeString=ht.NIFTI1.prototype.getUnitsCodeString;getQformMat=ht.NIFTI1.prototype.getQformMat;convertNiftiQFormToNiftiSForm=ht.NIFTI1.prototype.convertNiftiQFormToNiftiSForm;convertNiftiSFormToNEMA=ht.NIFTI1.prototype.convertNiftiSFormToNEMA;nifti_mat33_mul=ht.NIFTI1.prototype.nifti_mat33_mul;nifti_mat33_determ=ht.NIFTI1.prototype.nifti_mat33_determ;toArrayBuffer(t=!1){let i=544;if(t)for(let c of this.extensions)i+=c.esize;let s=new Uint8Array(i),o=new DataView(s.buffer);o.setInt32(0,540,this.littleEndian),s.set(Buffer.from(this.magic),4),o.setInt16(12,this.datatypeCode,this.littleEndian),o.setInt16(14,this.numBitsPerVoxel,this.littleEndian);for(let c=0;c<8;c++)o.setBigInt64(16+8*c,BigInt(this.dims[c]),this.littleEndian);o.setFloat64(80,this.intent_p1,this.littleEndian),o.setFloat64(88,this.intent_p2,this.littleEndian),o.setFloat64(96,this.intent_p3,this.littleEndian);for(let c=0;c<8;c++)o.setFloat64(104+8*c,this.pixDims[c],this.littleEndian);o.setBigInt64(168,BigInt(this.vox_offset),this.littleEndian),o.setFloat64(176,this.scl_slope,this.littleEndian),o.setFloat64(184,this.scl_inter,this.littleEndian),o.setFloat64(192,this.cal_max,this.littleEndian),o.setFloat64(200,this.cal_min,this.littleEndian),o.setFloat64(208,this.slice_duration,this.littleEndian),o.setFloat64(216,this.toffset,this.littleEndian),o.setBigInt64(224,BigInt(this.slice_start),this.littleEndian),o.setBigInt64(232,BigInt(this.slice_end),this.littleEndian),s.set(Buffer.from(this.description),240),s.set(Buffer.from(this.aux_file),320),o.setInt32(344,this.qform_code,this.littleEndian),o.setInt32(348,this.sform_code,this.littleEndian),o.setFloat64(352,this.quatern_b,this.littleEndian),o.setFloat64(360,this.quatern_c,this.littleEndian),o.setFloat64(368,this.quatern_d,this.littleEndian),o.setFloat64(376,this.qoffset_x,this.littleEndian),o.setFloat64(384,this.qoffset_y,this.littleEndian),o.setFloat64(392,this.qoffset_z,this.littleEndian);let a=this.affine.flat();for(let c=0;c<12;c++)o.setFloat64(400+8*c,a[c],this.littleEndian);if(o.setInt32(496,this.slice_code,this.littleEndian),o.setInt32(500,this.xyzt_units,this.littleEndian),o.setInt32(504,this.intent_code,this.littleEndian),s.set(Buffer.from(this.intent_name),508),o.setUint8(524,this.dim_info),t){s.set(Uint8Array.from([1,0,0,0]),540);let c=this.getExtensionLocation();for(let l of this.extensions)o.setInt32(c,l.esize,l.littleEndian),o.setInt32(c+4,l.ecode,l.littleEndian),s.set(new Uint8Array(l.edata),c+8),c+=l.esize}else s.set(new Uint8Array(4).fill(0),540);return s.buffer}};Wa.NIFTI2=Vf});var Rx=q(ae=>{"use strict";var Bk=ae&&ae.__createBinding||(Object.create?function(e,t,r,n){n===void 0&&(n=r);var i=Object.getOwnPropertyDescriptor(t,r);(!i||("get"in i?!t.__esModule:i.writable||i.configurable))&&(i={enumerable:!0,get:function(){return t[r]}}),Object.defineProperty(e,n,i)}:function(e,t,r,n){n===void 0&&(n=r),e[n]=t[r]}),zk=ae&&ae.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),$k=ae&&ae.__importStar||function(e){if(e&&e.__esModule)return e;var t={};if(e!=null)for(var r in e)r!=="default"&&Object.prototype.hasOwnProperty.call(e,r)&&Bk(t,e,r);return zk(t,e),t};Object.defineProperty(ae,"__esModule",{value:!0});ae.readExtensionData=ae.readExtension=ae.readImage=ae.hasExtension=ae.readHeader=ae.decompress=ae.isCompressed=ae.isNIFTI=ae.isNIFTI2=ae.isNIFTI1=ae.NIFTIEXTENSION=ae.Utils=ae.NIFTI2=ae.NIFTI1=void 0;var Vk=$k(Tx()),Ft=Ha(),lr=jf(),Nx=xs(),jk=Ha();Object.defineProperty(ae,"NIFTI1",{enumerable:!0,get:function(){return jk.NIFTI1}});var Gk=jf();Object.defineProperty(ae,"NIFTI2",{enumerable:!0,get:function(){return Gk.NIFTI2}});var qk=xs();Object.defineProperty(ae,"Utils",{enumerable:!0,get:function(){return qk.Utils}});var Kk=Bf();Object.defineProperty(ae,"NIFTIEXTENSION",{enumerable:!0,get:function(){return Kk.NIFTIEXTENSION}});function Gf(e,t=!1){var r,n,i,s;return e.byteLength<Ft.NIFTI1.STANDARD_HEADER_SIZE?!1:(r=new DataView(e),r&&(n=r.getUint8(Ft.NIFTI1.MAGIC_NUMBER_LOCATION)),i=r.getUint8(Ft.NIFTI1.MAGIC_NUMBER_LOCATION+1),s=r.getUint8(Ft.NIFTI1.MAGIC_NUMBER_LOCATION+2),t&&n===Ft.NIFTI1.MAGIC_NUMBER2[0]&&i===Ft.NIFTI1.MAGIC_NUMBER2[1]&&s===Ft.NIFTI1.MAGIC_NUMBER2[2]?!0:n===Ft.NIFTI1.MAGIC_NUMBER[0]&&i===Ft.NIFTI1.MAGIC_NUMBER[1]&&s===Ft.NIFTI1.MAGIC_NUMBER[2])}ae.isNIFTI1=Gf;function qf(e,t=!1){var r,n,i,s;return e.byteLength<Ft.NIFTI1.STANDARD_HEADER_SIZE?!1:(r=new DataView(e),n=r.getUint8(lr.NIFTI2.MAGIC_NUMBER_LOCATION),i=r.getUint8(lr.NIFTI2.MAGIC_NUMBER_LOCATION+1),s=r.getUint8(lr.NIFTI2.MAGIC_NUMBER_LOCATION+2),t&&n===lr.NIFTI2.MAGIC_NUMBER2[0]&&i===lr.NIFTI2.MAGIC_NUMBER2[1]&&s===lr.NIFTI2.MAGIC_NUMBER2[2]?!0:n===lr.NIFTI2.MAGIC_NUMBER[0]&&i===lr.NIFTI2.MAGIC_NUMBER[1]&&s===lr.NIFTI2.MAGIC_NUMBER[2])}ae.isNIFTI2=qf;function Yk(e,t=!1){return Gf(e,t)||qf(e,t)}ae.isNIFTI=Yk;function Dx(e){var t,r,n;return!!(e&&(t=new DataView(e),r=t.getUint8(0),n=t.getUint8(1),r===Nx.Utils.GUNZIP_MAGIC_COOKIE1||n===Nx.Utils.GUNZIP_MAGIC_COOKIE2))}ae.isCompressed=Dx;function Px(e){return Vk.decompressSync(new Uint8Array(e)).buffer}ae.decompress=Px;function Jk(e,t=!1){var r=null;return Dx(e)&&(e=Px(e)),Gf(e,t)?r=new Ft.NIFTI1:qf(e,t)&&(r=new lr.NIFTI2),r?r.readHeader(e):console.error("That file does not appear to be NIFTI!"),r}ae.readHeader=Jk;function Hk(e){return e.extensionFlag[0]!=0}ae.hasExtension=Hk;function Wk(e,t){var r=e.vox_offset,n=1,i=1;e.dims[4]&&(n=e.dims[4]),e.dims[5]&&(i=e.dims[5]);var s=e.dims[1]*e.dims[2]*e.dims[3]*n*i*(e.numBitsPerVoxel/8);return t.slice(r,r+s)}ae.readImage=Wk;function Xk(e,t){var r=e.getExtensionLocation(),n=e.extensionSize;return t.slice(r,r+n)}ae.readExtension=Xk;function Zk(e,t){var r=e.getExtensionLocation(),n=e.extensionSize;return t.slice(r+8,r+n)}ae.readExtensionData=Zk});var _S=q(Ah=>{var IS;(function(e){typeof DO_NOT_EXPORT_CRC>"u"?typeof Ah=="object"?e(Ah):typeof define=="function"&&define.amd?define(function(){var t={};return e(t),t}):e(IS={}):e(IS={})})(function(e){e.version="1.2.2";function t(){for(var v=0,w=new Array(256),x=0;x!=256;++x)v=x,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,w[x]=v;return typeof Int32Array<"u"?new Int32Array(w):w}var r=t();function n(v){var w=0,x=0,k=0,M=typeof Int32Array<"u"?new Int32Array(4096):new Array(4096);for(k=0;k!=256;++k)M[k]=v[k];for(k=0;k!=256;++k)for(x=v[k],w=256+k;w<4096;w+=256)x=M[w]=x>>>8^v[x&255];var P=[];for(k=1;k!=16;++k)P[k-1]=typeof Int32Array<"u"?M.subarray(k*256,k*256+256):M.slice(k*256,k*256+256);return P}var i=n(r),s=i[0],o=i[1],a=i[2],c=i[3],l=i[4],u=i[5],f=i[6],h=i[7],p=i[8],d=i[9],m=i[10],g=i[11],y=i[12],I=i[13],_=i[14];function E(v,w){for(var x=w^-1,k=0,M=v.length;k<M;)x=x>>>8^r[(x^v.charCodeAt(k++))&255];return~x}function b(v,w){for(var x=w^-1,k=v.length-15,M=0;M<k;)x=_[v[M++]^x&255]^I[v[M++]^x>>8&255]^y[v[M++]^x>>16&255]^g[v[M++]^x>>>24]^m[v[M++]]^d[v[M++]]^p[v[M++]]^h[v[M++]]^f[v[M++]]^u[v[M++]]^l[v[M++]]^c[v[M++]]^a[v[M++]]^o[v[M++]]^s[v[M++]]^r[v[M++]];for(k+=15;M<k;)x=x>>>8^r[(x^v[M++])&255];return~x}function C(v,w){for(var x=w^-1,k=0,M=v.length,P=0,F=0;k<M;)P=v.charCodeAt(k++),P<128?x=x>>>8^r[(x^P)&255]:P<2048?(x=x>>>8^r[(x^(192|P>>6&31))&255],x=x>>>8^r[(x^(128|P&63))&255]):P>=55296&&P<57344?(P=(P&1023)+64,F=v.charCodeAt(k++)&1023,x=x>>>8^r[(x^(240|P>>8&7))&255],x=x>>>8^r[(x^(128|P>>2&63))&255],x=x>>>8^r[(x^(128|F>>6&15|(P&3)<<4))&255],x=x>>>8^r[(x^(128|F&63))&255]):(x=x>>>8^r[(x^(224|P>>12&15))&255],x=x>>>8^r[(x^(128|P>>6&63))&255],x=x>>>8^r[(x^(128|P&63))&255]);return~x}e.table=r,e.bstr=E,e.buf=b,e.str=C})});var gw=q(jh=>{var mw;(function(e){typeof DO_NOT_EXPORT_CRC>"u"?typeof jh=="object"?e(jh):typeof define=="function"&&define.amd?define(function(){var t={};return e(t),t}):e(mw={}):e(mw={})})(function(e){e.version="1.2.2";function t(){for(var v=0,w=new Array(256),x=0;x!=256;++x)v=x,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,w[x]=v;return typeof Int32Array<"u"?new Int32Array(w):w}var r=t();function n(v){var w=0,x=0,k=0,M=typeof Int32Array<"u"?new Int32Array(4096):new Array(4096);for(k=0;k!=256;++k)M[k]=v[k];for(k=0;k!=256;++k)for(x=v[k],w=256+k;w<4096;w+=256)x=M[w]=x>>>8^v[x&255];var P=[];for(k=1;k!=16;++k)P[k-1]=typeof Int32Array<"u"?M.subarray(k*256,k*256+256):M.slice(k*256,k*256+256);return P}var i=n(r),s=i[0],o=i[1],a=i[2],c=i[3],l=i[4],u=i[5],f=i[6],h=i[7],p=i[8],d=i[9],m=i[10],g=i[11],y=i[12],I=i[13],_=i[14];function E(v,w){for(var x=w^-1,k=0,M=v.length;k<M;)x=x>>>8^r[(x^v.charCodeAt(k++))&255];return~x}function b(v,w){for(var x=w^-1,k=v.length-15,M=0;M<k;)x=_[v[M++]^x&255]^I[v[M++]^x>>8&255]^y[v[M++]^x>>16&255]^g[v[M++]^x>>>24]^m[v[M++]]^d[v[M++]]^p[v[M++]]^h[v[M++]]^f[v[M++]]^u[v[M++]]^l[v[M++]]^c[v[M++]]^a[v[M++]]^o[v[M++]]^s[v[M++]]^r[v[M++]];for(k+=15;M<k;)x=x>>>8^r[(x^v[M++])&255];return~x}function C(v,w){for(var x=w^-1,k=0,M=v.length,P=0,F=0;k<M;)P=v.charCodeAt(k++),P<128?x=x>>>8^r[(x^P)&255]:P<2048?(x=x>>>8^r[(x^(192|P>>6&31))&255],x=x>>>8^r[(x^(128|P&63))&255]):P>=55296&&P<57344?(P=(P&1023)+64,F=v.charCodeAt(k++)&1023,x=x>>>8^r[(x^(240|P>>8&7))&255],x=x>>>8^r[(x^(128|P>>2&63))&255],x=x>>>8^r[(x^(128|F>>6&15|(P&3)<<4))&255],x=x>>>8^r[(x^(128|F&63))&255]):(x=x>>>8^r[(x^(224|P>>12&15))&255],x=x>>>8^r[(x^(128|P>>6&63))&255],x=x>>>8^r[(x^(128|P&63))&255]);return~x}e.table=r,e.bstr=E,e.buf=b,e.str=C})});var gP=Ri(Ld(),1),yP=Ri(Gd(),1);var Pb=typeof global=="object"&&global&&global.Object===Object&&global,qd=Pb;var Rb=typeof self=="object"&&self&&self.Object===Object&&self,Ob=qd||Rb||Function("return this")(),Js=Ob;var Ub=Js.Symbol,Cn=Ub;var Kd=Object.prototype,Lb=Kd.hasOwnProperty,Fb=Kd.toString,zi=Cn?Cn.toStringTag:void 0;function Bb(e){var t=Lb.call(e,zi),r=e[zi];try{e[zi]=void 0;var n=!0}catch{}var i=Fb.call(e);return n&&(t?e[zi]=r:delete e[zi]),i}var Yd=Bb;var zb=Object.prototype,$b=zb.toString;function Vb(e){return $b.call(e)}var Jd=Vb;var jb="[object Null]",Gb="[object Undefined]",Hd=Cn?Cn.toStringTag:void 0;function qb(e){return e==null?e===void 0?Gb:jb:Hd&&Hd in Object(e)?Yd(e):Jd(e)}var Wd=qb;function Kb(e){return e!=null&&typeof e=="object"}var Xd=Kb;var Yb="[object Symbol]";function Jb(e){return typeof e=="symbol"||Xd(e)&&Wd(e)==Yb}var Zd=Jb;var Hb=/\s/;function Wb(e){for(var t=e.length;t--&&Hb.test(e.charAt(t)););return t}var Qd=Wb;var Xb=/^\s+/;function Zb(e){return e&&e.slice(0,Qd(e)+1).replace(Xb,"")}var em=Zb;function Qb(e){var t=typeof e;return e!=null&&(t=="object"||t=="function")}var en=Qb;var tm=NaN,eI=/^[-+]0x[0-9a-f]+$/i,tI=/^0b[01]+$/i,rI=/^0o[0-7]+$/i,nI=parseInt;function iI(e){if(typeof e=="number")return e;if(Zd(e))return tm;if(en(e)){var t=typeof e.valueOf=="function"?e.valueOf():e;e=en(t)?t+"":t}if(typeof e!="string")return e===0?e:+e;e=em(e);var r=tI.test(e);return r||rI.test(e)?nI(e.slice(2),r?2:8):eI.test(e)?tm:+e}var Sl=iI;var sI=function(){return Js.Date.now()},Hs=sI;var oI="Expected a function",aI=Math.max,cI=Math.min;function lI(e,t,r){var n,i,s,o,a,c,l=0,u=!1,f=!1,h=!0;if(typeof e!="function")throw new TypeError(oI);t=Sl(t)||0,en(r)&&(u=!!r.leading,f="maxWait"in r,s=f?aI(Sl(r.maxWait)||0,t):s,h="trailing"in r?!!r.trailing:h);function p(C){var v=n,w=i;return n=i=void 0,l=C,o=e.apply(w,v),o}function d(C){return l=C,a=setTimeout(y,t),u?p(C):o}function m(C){var v=C-c,w=C-l,x=t-v;return f?cI(x,s-w):x}function g(C){var v=C-c,w=C-l;return c===void 0||v>=t||v<0||f&&w>=s}function y(){var C=Hs();if(g(C))return I(C);a=setTimeout(y,m(C))}function I(C){return a=void 0,h&&n?p(C):(n=i=void 0,o)}function _(){a!==void 0&&clearTimeout(a),l=0,n=c=i=a=void 0}function E(){return a===void 0?o:I(Hs())}function b(){var C=Hs(),v=g(C);if(n=arguments,i=this,c=C,v){if(a===void 0)return d(c);if(f)return clearTimeout(a),a=setTimeout(y,t),p(c)}return a===void 0&&(a=setTimeout(y,t)),o}return b.cancel=_,b.flush=E,b}var tn=lI;var uI="Expected a function";function fI(e,t,r){var n=!0,i=!0;if(typeof e!="function")throw new TypeError(uI);return en(r)&&(n="leading"in r?!!r.leading:n,i="trailing"in r?!!r.trailing:i),tn(e,t,{leading:n,maxWait:t,trailing:i})}var Ws=fI;function hI(e){typeof e=="object"?e.dispose():e()}function wl(e){for(let t=e.length;t>0;--t)hI(e[t-1])}function pI(e,t,r,n){return e.addEventListener(t,r,n),()=>e.removeEventListener(t,r,n)}var be=class{refCount=1;wasDisposed;disposers;addRef(){return++this.refCount,this}disposedStacks;dispose(){--this.refCount===0&&this.refCountReachedZero()}[Symbol.dispose](){this.dispose()}refCountReachedZero(){this.disposed();let{disposers:t}=this;t!==void 0&&(wl(t),this.disposers=void 0),this.wasDisposed=!0}disposed(){}registerDisposer(t){let{disposers:r}=this;return r==null?this.disposers=[t]:r.push(t),t}unregisterDisposer(t){let{disposers:r}=this;if(r!=null){let n=r.indexOf(t);n!==-1&&r.splice(n,1)}return t}registerEventListener(t,r,n,i){this.registerDisposer(pI(t,r,n,i))}registerCancellable(t){return this.registerDisposer(()=>{t.cancel()}),t}},Xs=class extends be{constructor(t){super(),this.value=t}};var jt=class{handlers=new Set;count=0;constructor(){let t=this;this.dispatch=function(){++t.count,t.handlers.forEach(r=>{r.apply(this,arguments)})}}add(t){return this.handlers.add(t),()=>this.remove(t)}addOnce(t){let{handlers:r}=this;function n(...i){r.delete(n),t(...i)}r.add(n)}remove(t){return this.handlers.delete(t)}dispatch;dispose(){this.handlers=void 0}};var $e=class extends jt{};var Gt=class{constructor(t){this.value_=t}get value(){return this.value_}set value(t){t!==this.value_&&(this.value_=t,this.changed.dispatch())}changed=new $e};function rm(e,...t){let r=t.map(c=>c.value),n=t.length,i=new be,s=e(i,...r),o=tn(()=>{let c=!1;for(let l=0;l<n;++l){let f=t[l].value;r[l]!==f&&(r[l]=f,c=!0)}c&&(i.dispose(),i=new be,s=e(i,...r))},0),a=t.map(c=>c.changed.add(o));return{flush(){o.flush()},dispose(){o.cancel(),wl(a),i.dispose()},get value(){return o.flush(),s}}}function nm(e,t){if(e===void 0)return;if(e.aborted){t(e.reason);return}function r(){t(this.reason)}return e.addEventListener("abort",r,{once:!0}),{[Symbol.dispose](){e.removeEventListener("abort",r)}}}var Zs=class{consumers=new Map;controller=new AbortController;retainCount=0;get signal(){return this.controller.signal}addConsumer(t){if(!this.controller.signal.aborted){if(t!==void 0){let n=function(){i.consumers.delete(n),--i.retainCount===0&&(i.controller.abort(),i[Symbol.dispose]())};var r=n;if(t.aborted)return;let i=this;t.addEventListener("abort",n,{once:!0})}++this.retainCount}}[Symbol.dispose](){for(let[t,r]of this.consumers)r.removeEventListener("abort",t);this.consumers.clear(),this.retainCount=0}start(){this.retainCount===0&&this.controller.abort()}};function im(e,t){let{promise:r,resolve:n,reject:i}=Promise.withResolvers(),s=nm(e,t);return{promise:r,resolve:o=>{s?.[Symbol.dispose](),n(o)},reject:o=>{s?.[Symbol.dispose](),i(o)}}}function Qs(e,t){return t===void 0?e:t.aborted?Promise.reject(t.reason):new Promise((r,n)=>{let i=nm(t,s=>{n(s)});e.then(s=>{i?.[Symbol.dispose](),r(s)},s=>{i?.[Symbol.dispose](),n(s)})})}var He=class{constructor(t,r){this.listener=t;let{id:n=Math.random(),startTime:i=Date.now(),message:s}=r;this.id=n,this.startTime=i,this.message=s,t.addSpan(this)}id;startTime;message;[Symbol.dispose](){this.listener.removeSpan(this.id)}},El=class{items=new Map;add(t){let{items:r}=this,n=(r.get(t)??0)+1;return r.set(t,n),n}delete(t){let{items:r}=this,n=r.get(t);return n>1?(n-=1,r.set(t,n),n):(r.delete(t),0)}has(t){return this.items.has(t)}keys(){return this.items.keys()}entries(){return this.items.entries()}[Symbol.iterator](){return this.items.keys()}},bl=class{constructor(t){this.getKey=t}items=new Map;add(t){let{items:r}=this,n=this.getKey(t),i=r.get(n);return i===void 0?(r.set(n,{value:t,count:1}),1):i.count+=1}delete(t){return this.deleteKey(this.getKey(t))}deleteKey(t){let{items:r}=this,n=r.get(t);return n!==void 0&&n.count>1?n.count-=1:(r.delete(t),0)}has(t){return this.items.has(this.getKey(t))}*[Symbol.iterator](){for(let t of this.items.values())yield t.value}};function dI(e){return e.id}var Il=class extends bl{constructor(){super(dI)}},eo=class{spans=new Il;listeners=new El;addSpan(t){if(this.spans.add(t)===1)for(let r of this.listeners)r.addSpan(t)}removeSpan(t){if(this.spans.deleteKey(t)===0)for(let r of this.listeners)r.removeSpan(t)}addListener(t){if(t!==void 0&&this.listeners.add(t)===1)for(let r of this.spans)t.addSpan(r)}removeListener(t){if(t!==void 0&&this.listeners.delete(t)===0)for(let r of this.spans)t.removeSpan(r.id)}};var ro=!(typeof Window<"u"&&self instanceof Window),om=!1,sm=!1,_l="rpc.promise.response",am="rpc.promise.cancel",cm="rpc.promise.addProgressSpan",lm="rpc.promise.removeProgressSpan",um="rpc.ready",Cl=new Map;function X(e,t){Cl.set(e,t)}var Al=class{constructor(t,r){this.rpc=t,this.id=r}addSpan(t){this.rpc.invoke(cm,{id:this.id,span:{id:t.id,message:t.message,startTime:t.startTime}})}removeSpan(t){this.rpc.invoke(lm,{id:this.id,spanId:t})}};function xt(e,t){X(e,function(r){let n=r.id,i=new AbortController,s;r.progressListener===!0&&(s=new Al(this,n));let o=t.call(this,r,{signal:i.signal,progressListener:s});this.set(n,{promise:o,abortController:i}),o.then(({value:a,transfers:c})=>{this.delete(n),this.invoke(_l,{id:n,value:a},c)},a=>{this.delete(n),this.invoke(_l,{id:n,error:a})})})}X(am,function(e){let t=e.id,r=this.get(t);if(r!==void 0){let{abortController:n}=r;n.abort()}});X(_l,function(e){let t=e.id,{resolve:r,reject:n}=this.get(t);this.delete(t),Object.prototype.hasOwnProperty.call(e,"value")?r(e.value):n(e.error)});X(cm,function(e){let t=e.id,{progressListener:r}=this.get(t);new He(r,e.span)});X(lm,function(e){let t=e.id,{progressListener:r}=this.get(t);r.removeSpan(e.spanId)});X(um,function(e){this.onPeerReady()});var mI=ro?-1:0,to=class{constructor(t,r){this.target=t,r&&(this.queue=[]),t.onmessage=n=>{let i=n.data;if(sm&&console.log("Received message",i),Cl.get(i.functionName)===void 0)throw new Error(`Missing RPC function: ${i.functionName}`);Cl.get(i.functionName).call(this,i)}}objects=new Map;nextId=mI;queue;sendReady(){this.invoke(um,{})}onPeerReady(){let{queue:t}=this;if(t!==void 0){this.queue=void 0;for(let{data:r,transfers:n}of t)this.target.postMessage(r,n)}}get numObjects(){return this.objects.size}set(t,r){this.objects.set(t,r)}delete(t){this.objects.delete(t)}get(t){return this.objects.get(t)}getRef(t){let r=t.id,n=this.get(r);return n.referencedGeneration=t.gen,n.addRef(),n}getOptionalRef(t){if(t===void 0)return;let r=t.id,n=this.get(r);return n.referencedGeneration=t.gen,n.addRef(),n}invoke(t,r,n){r.functionName=t,sm&&console.trace("Sending message",r);let{queue:i}=this;if(i!==void 0){i.push({data:r,transfers:n});return}this.target.postMessage(r,n)}promiseInvoke(t,r,n){let i,s,o;if(n!==void 0&&({signal:i,progressListener:s,transfers:o}=n),i?.aborted)return Promise.reject(i.reason);s!==void 0&&(r.progressListener=!0);let a=r.id=this.newId();this.invoke(t,r,o);let{promise:c,resolve:l,reject:u}=i===void 0?Promise.withResolvers():im(i,()=>{this.invoke(am,{id:a})});return this.set(a,{resolve:l,reject:u,progressListener:s}),c}newId(){return ro?this.nextId--:this.nextId++}},rn=class extends be{rpc=null;rpcId=null;isOwner;unreferencedGeneration;referencedGeneration;initializeSharedObject(t,r=t.newId()){this.rpc=t,this.rpcId=r,this.isOwner=!1,t.set(r,this)}initializeCounterpart(t,r={}){this.initializeSharedObject(t),this.unreferencedGeneration=0,this.referencedGeneration=0,this.isOwner=!0,r.id=this.rpcId,r.type=this.RPC_TYPE_ID,t.invoke("SharedObject.new",r)}dispose(){super.dispose()}addCounterpartRef(){return{id:this.rpcId,gen:++this.referencedGeneration}}refCountReachedZero(){this.isOwner===!0?this.referencedGeneration===this.unreferencedGeneration&&this.ownerDispose():this.isOwner===!1?this.rpc.invoke("SharedObject.refCountReachedZero",{id:this.rpcId,gen:this.referencedGeneration}):super.refCountReachedZero()}ownerDispose(){om&&console.log(`[${ro}] #rpc object = ${this.rpc.numObjects}`);let{rpc:t,rpcId:r}=this;super.refCountReachedZero(),t.delete(r),t.invoke("SharedObject.dispose",{id:r})}counterpartRefCountReachedZero(t){this.unreferencedGeneration=t,this.refCount===0&&t===this.referencedGeneration&&this.ownerDispose()}};function Ml(e,t,r={}){t!=null&&e.initializeSharedObject(t,r.id)}var me=class extends rn{constructor(t,r={}){super(),Ml(this,t,r)}};X("SharedObject.dispose",function(e){let t=this.get(e.id);if(t.refCount!==0)throw new Error("Attempted to dispose object with non-zero reference count.");om&&console.log(`[${ro}] #rpc objects: ${this.numObjects}`),t.disposed(),this.delete(t.rpcId),t.rpcId=null,t.rpc=null});X("SharedObject.refCountReachedZero",function(e){let t=this.get(e.id),r=e.gen;t.counterpartRefCountReachedZero(r)});var fm=new Map;function hm(e){return t=>{t.prototype.RPC_TYPE_ID=e}}function j(e){return t=>{if(e!==void 0)t.prototype.RPC_TYPE_ID=e;else if(e=t.prototype.RPC_TYPE_ID,e===void 0)throw new Error("RPC_TYPE_ID should have already been defined");fm.set(e,t)}}X("SharedObject.new",function(e){let t=this,r=e.type,n=fm.get(r),i=new n(t,e);--i.refCount});var gI=Object.defineProperty,yI=Object.getOwnPropertyDescriptor,vI=(e,t,r,n)=>{for(var i=n>1?void 0:n?yI(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&gI(t,r,i),i};var pm="SharedWatchableValue.changed",no=class extends me{base;updatingValue_=!1;constructor(e,t={}){super(e,t),e!==void 0&&(this.base=new Gt(t.value),this.setupChangedHandler())}initializeCounterpart(e,t={}){t.value=this.value,super.initializeCounterpart(e,t)}setupChangedHandler(){this.registerDisposer(this.base.changed.add(()=>{if(this.updatingValue_)this.updatingValue_=!1;else{let{rpc:e}=this;e!==null&&e.invoke(pm,{id:this.rpcId,value:this.value})}}))}static makeFromExisting(e,t){let r=new no;return r.base=t,r.setupChangedHandler(),r.initializeCounterpart(e),r}static make(e,t){return no.makeFromExisting(e,new Gt(t))}get value(){return this.base.value}set value(e){this.base.value=e}get changed(){return this.base.changed}};no=vI([j("SharedWatchableValue")],no);X(pm,function(e){let t=this.get(e.id);t.updatingValue_=!0,t.base.value=e.value,t.updatingValue_=!1});var z=(e=>(e[e.GPU_MEMORY=0]="GPU_MEMORY",e[e.SYSTEM_MEMORY=1]="SYSTEM_MEMORY",e[e.SYSTEM_MEMORY_WORKER=2]="SYSTEM_MEMORY_WORKER",e[e.DOWNLOADING=3]="DOWNLOADING",e[e.QUEUED=4]="QUEUED",e[e.NEW=5]="NEW",e[e.FAILED=6]="FAILED",e[e.EXPIRED=7]="EXPIRED",e))(z||{}),dm=8,se=(e=>(e[e.FIRST_TIER=0]="FIRST_TIER",e[e.FIRST_ORDERED_TIER=0]="FIRST_ORDERED_TIER",e[e.VISIBLE=0]="VISIBLE",e[e.PREFETCH=1]="PREFETCH",e[e.LAST_ORDERED_TIER=1]="LAST_ORDERED_TIER",e[e.RECENT=2]="RECENT",e[e.LAST_TIER=2]="LAST_TIER",e))(se||{}),kl=3,io=(e=>(e[e.totalTime=0]="totalTime",e[e.totalChunks=1]="totalChunks",e))(io||{}),$i=(e=>(e[e.numChunks=0]="numChunks",e[e.systemMemoryBytes=1]="systemMemoryBytes",e[e.gpuMemoryBytes=2]="gpuMemoryBytes",e))($i||{}),An=3,xI=2,mm=dm*kl*An+xI;function gm(e,t){return e*kl+t}function Tl(e){return dm*kl*An+e}var ym=1e13,vm="ChunkQueueManager",xm="ChunkManager",Sm="ChunkSource.invalidate",wm="ChunkQueueManager.requestChunkStatistics",Em="ChunkManager.chunkLayerStatistics";function so(e){let{next:t,prev:r}=e;return{insertAfter(n,i){let s=n[t];i[t]=s,i[r]=n,n[t]=i,s[r]=i},insertBefore(n,i){let s=n[r];i[r]=s,i[t]=n,n[r]=i,s[t]=i},front(n){let i=n[t];return i===n?null:i},back(n){let i=n[r];return i===n?null:i},pop(n){let i=n[t],s=n[r];return i[r]=s,s[t]=i,n[t]=null,n[r]=null,n},*iterator(n){for(let i=n[t];i!==n;i=i[t])yield i},*reverseIterator(n){for(let i=n[r];i!==n;i=i[r])yield i},initializeHead(n){n[t]=n[r]=n}}}function mr(e,t){return e<t?-1:e>t?1:0}function SI(e,t){return BigInt(e)|BigInt(t)<<32n}function Nl(){let e=Math.random()*4294967296>>>0,t=Math.random()*4294967296>>>0;return SI(e,t)}var Dl=0xffffffffffffffffn;var Z=1e-6,Se=typeof Float32Array<"u"?Float32Array:Array,Mt=Math.random;var SR=Math.PI/180;Math.hypot||(Math.hypot=function(){for(var e=0,t=arguments.length;t--;)e+=arguments[t]*arguments[t];return Math.sqrt(e)});var kt={};Pi(kt,{add:()=>jI,adjoint:()=>kI,clone:()=>EI,copy:()=>bI,create:()=>Pl,determinant:()=>TI,equals:()=>YI,exactEquals:()=>KI,frob:()=>VI,fromMat2d:()=>LI,fromMat4:()=>wI,fromQuat:()=>FI,fromRotation:()=>OI,fromScaling:()=>UI,fromTranslation:()=>RI,fromValues:()=>II,identity:()=>CI,invert:()=>MI,mul:()=>JI,multiply:()=>bm,multiplyScalar:()=>GI,multiplyScalarAndAdd:()=>qI,normalFromMat4:()=>BI,projection:()=>zI,rotate:()=>DI,scale:()=>PI,set:()=>_I,str:()=>$I,sub:()=>HI,subtract:()=>Im,translate:()=>NI,transpose:()=>AI});function Pl(){var e=new Se(9);return Se!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[5]=0,e[6]=0,e[7]=0),e[0]=1,e[4]=1,e[8]=1,e}function wI(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[4],e[4]=t[5],e[5]=t[6],e[6]=t[8],e[7]=t[9],e[8]=t[10],e}function EI(e){var t=new Se(9);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t[4]=e[4],t[5]=e[5],t[6]=e[6],t[7]=e[7],t[8]=e[8],t}function bI(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[8]=t[8],e}function II(e,t,r,n,i,s,o,a,c){var l=new Se(9);return l[0]=e,l[1]=t,l[2]=r,l[3]=n,l[4]=i,l[5]=s,l[6]=o,l[7]=a,l[8]=c,l}function _I(e,t,r,n,i,s,o,a,c,l){return e[0]=t,e[1]=r,e[2]=n,e[3]=i,e[4]=s,e[5]=o,e[6]=a,e[7]=c,e[8]=l,e}function CI(e){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=1,e[5]=0,e[6]=0,e[7]=0,e[8]=1,e}function AI(e,t){if(e===t){var r=t[1],n=t[2],i=t[5];e[1]=t[3],e[2]=t[6],e[3]=r,e[5]=t[7],e[6]=n,e[7]=i}else e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8];return e}function MI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],l=t[7],u=t[8],f=u*o-a*l,h=-u*s+a*c,p=l*s-o*c,d=r*f+n*h+i*p;return d?(d=1/d,e[0]=f*d,e[1]=(-u*n+i*l)*d,e[2]=(a*n-i*o)*d,e[3]=h*d,e[4]=(u*r-i*c)*d,e[5]=(-a*r+i*s)*d,e[6]=p*d,e[7]=(-l*r+n*c)*d,e[8]=(o*r-n*s)*d,e):null}function kI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],l=t[7],u=t[8];return e[0]=o*u-a*l,e[1]=i*l-n*u,e[2]=n*a-i*o,e[3]=a*c-s*u,e[4]=r*u-i*c,e[5]=i*s-r*a,e[6]=s*l-o*c,e[7]=n*c-r*l,e[8]=r*o-n*s,e}function TI(e){var t=e[0],r=e[1],n=e[2],i=e[3],s=e[4],o=e[5],a=e[6],c=e[7],l=e[8];return t*(l*s-o*c)+r*(-l*i+o*a)+n*(c*i-s*a)}function bm(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],l=t[6],u=t[7],f=t[8],h=r[0],p=r[1],d=r[2],m=r[3],g=r[4],y=r[5],I=r[6],_=r[7],E=r[8];return e[0]=h*n+p*o+d*l,e[1]=h*i+p*a+d*u,e[2]=h*s+p*c+d*f,e[3]=m*n+g*o+y*l,e[4]=m*i+g*a+y*u,e[5]=m*s+g*c+y*f,e[6]=I*n+_*o+E*l,e[7]=I*i+_*a+E*u,e[8]=I*s+_*c+E*f,e}function NI(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],l=t[6],u=t[7],f=t[8],h=r[0],p=r[1];return e[0]=n,e[1]=i,e[2]=s,e[3]=o,e[4]=a,e[5]=c,e[6]=h*n+p*o+l,e[7]=h*i+p*a+u,e[8]=h*s+p*c+f,e}function DI(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],l=t[6],u=t[7],f=t[8],h=Math.sin(r),p=Math.cos(r);return e[0]=p*n+h*o,e[1]=p*i+h*a,e[2]=p*s+h*c,e[3]=p*o-h*n,e[4]=p*a-h*i,e[5]=p*c-h*s,e[6]=l,e[7]=u,e[8]=f,e}function PI(e,t,r){var n=r[0],i=r[1];return e[0]=n*t[0],e[1]=n*t[1],e[2]=n*t[2],e[3]=i*t[3],e[4]=i*t[4],e[5]=i*t[5],e[6]=t[6],e[7]=t[7],e[8]=t[8],e}function RI(e,t){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=1,e[5]=0,e[6]=t[0],e[7]=t[1],e[8]=1,e}function OI(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=n,e[1]=r,e[2]=0,e[3]=-r,e[4]=n,e[5]=0,e[6]=0,e[7]=0,e[8]=1,e}function UI(e,t){return e[0]=t[0],e[1]=0,e[2]=0,e[3]=0,e[4]=t[1],e[5]=0,e[6]=0,e[7]=0,e[8]=1,e}function LI(e,t){return e[0]=t[0],e[1]=t[1],e[2]=0,e[3]=t[2],e[4]=t[3],e[5]=0,e[6]=t[4],e[7]=t[5],e[8]=1,e}function FI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r+r,a=n+n,c=i+i,l=r*o,u=n*o,f=n*a,h=i*o,p=i*a,d=i*c,m=s*o,g=s*a,y=s*c;return e[0]=1-f-d,e[3]=u-y,e[6]=h+g,e[1]=u+y,e[4]=1-l-d,e[7]=p-m,e[2]=h-g,e[5]=p+m,e[8]=1-l-f,e}function BI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],l=t[7],u=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15],I=r*a-n*o,_=r*c-i*o,E=r*l-s*o,b=n*c-i*a,C=n*l-s*a,v=i*l-s*c,w=u*m-f*d,x=u*g-h*d,k=u*y-p*d,M=f*g-h*m,P=f*y-p*m,F=h*y-p*g,S=I*F-_*P+E*M+b*k-C*x+v*w;return S?(S=1/S,e[0]=(a*F-c*P+l*M)*S,e[1]=(c*k-o*F-l*x)*S,e[2]=(o*P-a*k+l*w)*S,e[3]=(i*P-n*F-s*M)*S,e[4]=(r*F-i*k+s*x)*S,e[5]=(n*k-r*P-s*w)*S,e[6]=(m*v-g*C+y*b)*S,e[7]=(g*E-d*v-y*_)*S,e[8]=(d*C-m*E+y*I)*S,e):null}function zI(e,t,r){return e[0]=2/t,e[1]=0,e[2]=0,e[3]=0,e[4]=-2/r,e[5]=0,e[6]=-1,e[7]=1,e[8]=1,e}function $I(e){return"mat3("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+", "+e[4]+", "+e[5]+", "+e[6]+", "+e[7]+", "+e[8]+")"}function VI(e){return Math.hypot(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8])}function jI(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e[3]=t[3]+r[3],e[4]=t[4]+r[4],e[5]=t[5]+r[5],e[6]=t[6]+r[6],e[7]=t[7]+r[7],e[8]=t[8]+r[8],e}function Im(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e[3]=t[3]-r[3],e[4]=t[4]-r[4],e[5]=t[5]-r[5],e[6]=t[6]-r[6],e[7]=t[7]-r[7],e[8]=t[8]-r[8],e}function GI(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e[3]=t[3]*r,e[4]=t[4]*r,e[5]=t[5]*r,e[6]=t[6]*r,e[7]=t[7]*r,e[8]=t[8]*r,e}function qI(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e[3]=t[3]+r[3]*n,e[4]=t[4]+r[4]*n,e[5]=t[5]+r[5]*n,e[6]=t[6]+r[6]*n,e[7]=t[7]+r[7]*n,e[8]=t[8]+r[8]*n,e}function KI(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]&&e[3]===t[3]&&e[4]===t[4]&&e[5]===t[5]&&e[6]===t[6]&&e[7]===t[7]&&e[8]===t[8]}function YI(e,t){var r=e[0],n=e[1],i=e[2],s=e[3],o=e[4],a=e[5],c=e[6],l=e[7],u=e[8],f=t[0],h=t[1],p=t[2],d=t[3],m=t[4],g=t[5],y=t[6],I=t[7],_=t[8];return Math.abs(r-f)<=Z*Math.max(1,Math.abs(r),Math.abs(f))&&Math.abs(n-h)<=Z*Math.max(1,Math.abs(n),Math.abs(h))&&Math.abs(i-p)<=Z*Math.max(1,Math.abs(i),Math.abs(p))&&Math.abs(s-d)<=Z*Math.max(1,Math.abs(s),Math.abs(d))&&Math.abs(o-m)<=Z*Math.max(1,Math.abs(o),Math.abs(m))&&Math.abs(a-g)<=Z*Math.max(1,Math.abs(a),Math.abs(g))&&Math.abs(c-y)<=Z*Math.max(1,Math.abs(c),Math.abs(y))&&Math.abs(l-I)<=Z*Math.max(1,Math.abs(l),Math.abs(I))&&Math.abs(u-_)<=Z*Math.max(1,Math.abs(u),Math.abs(_))}var JI=bm,HI=Im;var ge={};Pi(ge,{add:()=>N2,adjoint:()=>n2,clone:()=>XI,copy:()=>ZI,create:()=>WI,determinant:()=>i2,equals:()=>O2,exactEquals:()=>R2,frob:()=>T2,fromQuat:()=>E2,fromQuat2:()=>y2,fromRotation:()=>p2,fromRotationTranslation:()=>Am,fromRotationTranslationScale:()=>S2,fromRotationTranslationScaleOrigin:()=>w2,fromScaling:()=>h2,fromTranslation:()=>f2,fromValues:()=>QI,fromXRotation:()=>d2,fromYRotation:()=>m2,fromZRotation:()=>g2,frustum:()=>b2,getRotation:()=>x2,getScaling:()=>Mm,getTranslation:()=>v2,identity:()=>_m,invert:()=>r2,lookAt:()=>A2,mul:()=>U2,multiply:()=>Cm,multiplyScalar:()=>D2,multiplyScalarAndAdd:()=>P2,ortho:()=>C2,perspective:()=>I2,perspectiveFromFieldOfView:()=>_2,rotate:()=>a2,rotateX:()=>c2,rotateY:()=>l2,rotateZ:()=>u2,scale:()=>o2,set:()=>e2,str:()=>k2,sub:()=>L2,subtract:()=>km,targetTo:()=>M2,translate:()=>s2,transpose:()=>t2});function WI(){var e=new Se(16);return Se!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0),e[0]=1,e[5]=1,e[10]=1,e[15]=1,e}function XI(e){var t=new Se(16);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t[4]=e[4],t[5]=e[5],t[6]=e[6],t[7]=e[7],t[8]=e[8],t[9]=e[9],t[10]=e[10],t[11]=e[11],t[12]=e[12],t[13]=e[13],t[14]=e[14],t[15]=e[15],t}function ZI(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[8]=t[8],e[9]=t[9],e[10]=t[10],e[11]=t[11],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15],e}function QI(e,t,r,n,i,s,o,a,c,l,u,f,h,p,d,m){var g=new Se(16);return g[0]=e,g[1]=t,g[2]=r,g[3]=n,g[4]=i,g[5]=s,g[6]=o,g[7]=a,g[8]=c,g[9]=l,g[10]=u,g[11]=f,g[12]=h,g[13]=p,g[14]=d,g[15]=m,g}function e2(e,t,r,n,i,s,o,a,c,l,u,f,h,p,d,m,g){return e[0]=t,e[1]=r,e[2]=n,e[3]=i,e[4]=s,e[5]=o,e[6]=a,e[7]=c,e[8]=l,e[9]=u,e[10]=f,e[11]=h,e[12]=p,e[13]=d,e[14]=m,e[15]=g,e}function _m(e){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function t2(e,t){if(e===t){var r=t[1],n=t[2],i=t[3],s=t[6],o=t[7],a=t[11];e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=r,e[6]=t[9],e[7]=t[13],e[8]=n,e[9]=s,e[11]=t[14],e[12]=i,e[13]=o,e[14]=a}else e[0]=t[0],e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=t[1],e[5]=t[5],e[6]=t[9],e[7]=t[13],e[8]=t[2],e[9]=t[6],e[10]=t[10],e[11]=t[14],e[12]=t[3],e[13]=t[7],e[14]=t[11],e[15]=t[15];return e}function r2(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],l=t[7],u=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15],I=r*a-n*o,_=r*c-i*o,E=r*l-s*o,b=n*c-i*a,C=n*l-s*a,v=i*l-s*c,w=u*m-f*d,x=u*g-h*d,k=u*y-p*d,M=f*g-h*m,P=f*y-p*m,F=h*y-p*g,S=I*F-_*P+E*M+b*k-C*x+v*w;return S?(S=1/S,e[0]=(a*F-c*P+l*M)*S,e[1]=(i*P-n*F-s*M)*S,e[2]=(m*v-g*C+y*b)*S,e[3]=(h*C-f*v-p*b)*S,e[4]=(c*k-o*F-l*x)*S,e[5]=(r*F-i*k+s*x)*S,e[6]=(g*E-d*v-y*_)*S,e[7]=(u*v-h*E+p*_)*S,e[8]=(o*P-a*k+l*w)*S,e[9]=(n*k-r*P-s*w)*S,e[10]=(d*C-m*E+y*I)*S,e[11]=(f*E-u*C-p*I)*S,e[12]=(a*x-o*M-c*w)*S,e[13]=(r*M-n*x+i*w)*S,e[14]=(m*_-d*b-g*I)*S,e[15]=(u*b-f*_+h*I)*S,e):null}function n2(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],l=t[7],u=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15];return e[0]=a*(h*y-p*g)-f*(c*y-l*g)+m*(c*p-l*h),e[1]=-(n*(h*y-p*g)-f*(i*y-s*g)+m*(i*p-s*h)),e[2]=n*(c*y-l*g)-a*(i*y-s*g)+m*(i*l-s*c),e[3]=-(n*(c*p-l*h)-a*(i*p-s*h)+f*(i*l-s*c)),e[4]=-(o*(h*y-p*g)-u*(c*y-l*g)+d*(c*p-l*h)),e[5]=r*(h*y-p*g)-u*(i*y-s*g)+d*(i*p-s*h),e[6]=-(r*(c*y-l*g)-o*(i*y-s*g)+d*(i*l-s*c)),e[7]=r*(c*p-l*h)-o*(i*p-s*h)+u*(i*l-s*c),e[8]=o*(f*y-p*m)-u*(a*y-l*m)+d*(a*p-l*f),e[9]=-(r*(f*y-p*m)-u*(n*y-s*m)+d*(n*p-s*f)),e[10]=r*(a*y-l*m)-o*(n*y-s*m)+d*(n*l-s*a),e[11]=-(r*(a*p-l*f)-o*(n*p-s*f)+u*(n*l-s*a)),e[12]=-(o*(f*g-h*m)-u*(a*g-c*m)+d*(a*h-c*f)),e[13]=r*(f*g-h*m)-u*(n*g-i*m)+d*(n*h-i*f),e[14]=-(r*(a*g-c*m)-o*(n*g-i*m)+d*(n*c-i*a)),e[15]=r*(a*h-c*f)-o*(n*h-i*f)+u*(n*c-i*a),e}function i2(e){var t=e[0],r=e[1],n=e[2],i=e[3],s=e[4],o=e[5],a=e[6],c=e[7],l=e[8],u=e[9],f=e[10],h=e[11],p=e[12],d=e[13],m=e[14],g=e[15],y=t*o-r*s,I=t*a-n*s,_=t*c-i*s,E=r*a-n*o,b=r*c-i*o,C=n*c-i*a,v=l*d-u*p,w=l*m-f*p,x=l*g-h*p,k=u*m-f*d,M=u*g-h*d,P=f*g-h*m;return y*P-I*M+_*k+E*x-b*w+C*v}function Cm(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],l=t[6],u=t[7],f=t[8],h=t[9],p=t[10],d=t[11],m=t[12],g=t[13],y=t[14],I=t[15],_=r[0],E=r[1],b=r[2],C=r[3];return e[0]=_*n+E*a+b*f+C*m,e[1]=_*i+E*c+b*h+C*g,e[2]=_*s+E*l+b*p+C*y,e[3]=_*o+E*u+b*d+C*I,_=r[4],E=r[5],b=r[6],C=r[7],e[4]=_*n+E*a+b*f+C*m,e[5]=_*i+E*c+b*h+C*g,e[6]=_*s+E*l+b*p+C*y,e[7]=_*o+E*u+b*d+C*I,_=r[8],E=r[9],b=r[10],C=r[11],e[8]=_*n+E*a+b*f+C*m,e[9]=_*i+E*c+b*h+C*g,e[10]=_*s+E*l+b*p+C*y,e[11]=_*o+E*u+b*d+C*I,_=r[12],E=r[13],b=r[14],C=r[15],e[12]=_*n+E*a+b*f+C*m,e[13]=_*i+E*c+b*h+C*g,e[14]=_*s+E*l+b*p+C*y,e[15]=_*o+E*u+b*d+C*I,e}function s2(e,t,r){var n=r[0],i=r[1],s=r[2],o,a,c,l,u,f,h,p,d,m,g,y;return t===e?(e[12]=t[0]*n+t[4]*i+t[8]*s+t[12],e[13]=t[1]*n+t[5]*i+t[9]*s+t[13],e[14]=t[2]*n+t[6]*i+t[10]*s+t[14],e[15]=t[3]*n+t[7]*i+t[11]*s+t[15]):(o=t[0],a=t[1],c=t[2],l=t[3],u=t[4],f=t[5],h=t[6],p=t[7],d=t[8],m=t[9],g=t[10],y=t[11],e[0]=o,e[1]=a,e[2]=c,e[3]=l,e[4]=u,e[5]=f,e[6]=h,e[7]=p,e[8]=d,e[9]=m,e[10]=g,e[11]=y,e[12]=o*n+u*i+d*s+t[12],e[13]=a*n+f*i+m*s+t[13],e[14]=c*n+h*i+g*s+t[14],e[15]=l*n+p*i+y*s+t[15]),e}function o2(e,t,r){var n=r[0],i=r[1],s=r[2];return e[0]=t[0]*n,e[1]=t[1]*n,e[2]=t[2]*n,e[3]=t[3]*n,e[4]=t[4]*i,e[5]=t[5]*i,e[6]=t[6]*i,e[7]=t[7]*i,e[8]=t[8]*s,e[9]=t[9]*s,e[10]=t[10]*s,e[11]=t[11]*s,e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15],e}function a2(e,t,r,n){var i=n[0],s=n[1],o=n[2],a=Math.hypot(i,s,o),c,l,u,f,h,p,d,m,g,y,I,_,E,b,C,v,w,x,k,M,P,F,S,O;return a<Z?null:(a=1/a,i*=a,s*=a,o*=a,c=Math.sin(r),l=Math.cos(r),u=1-l,f=t[0],h=t[1],p=t[2],d=t[3],m=t[4],g=t[5],y=t[6],I=t[7],_=t[8],E=t[9],b=t[10],C=t[11],v=i*i*u+l,w=s*i*u+o*c,x=o*i*u-s*c,k=i*s*u-o*c,M=s*s*u+l,P=o*s*u+i*c,F=i*o*u+s*c,S=s*o*u-i*c,O=o*o*u+l,e[0]=f*v+m*w+_*x,e[1]=h*v+g*w+E*x,e[2]=p*v+y*w+b*x,e[3]=d*v+I*w+C*x,e[4]=f*k+m*M+_*P,e[5]=h*k+g*M+E*P,e[6]=p*k+y*M+b*P,e[7]=d*k+I*M+C*P,e[8]=f*F+m*S+_*O,e[9]=h*F+g*S+E*O,e[10]=p*F+y*S+b*O,e[11]=d*F+I*S+C*O,t!==e&&(e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e)}function c2(e,t,r){var n=Math.sin(r),i=Math.cos(r),s=t[4],o=t[5],a=t[6],c=t[7],l=t[8],u=t[9],f=t[10],h=t[11];return t!==e&&(e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[4]=s*i+l*n,e[5]=o*i+u*n,e[6]=a*i+f*n,e[7]=c*i+h*n,e[8]=l*i-s*n,e[9]=u*i-o*n,e[10]=f*i-a*n,e[11]=h*i-c*n,e}function l2(e,t,r){var n=Math.sin(r),i=Math.cos(r),s=t[0],o=t[1],a=t[2],c=t[3],l=t[8],u=t[9],f=t[10],h=t[11];return t!==e&&(e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*i-l*n,e[1]=o*i-u*n,e[2]=a*i-f*n,e[3]=c*i-h*n,e[8]=s*n+l*i,e[9]=o*n+u*i,e[10]=a*n+f*i,e[11]=c*n+h*i,e}function u2(e,t,r){var n=Math.sin(r),i=Math.cos(r),s=t[0],o=t[1],a=t[2],c=t[3],l=t[4],u=t[5],f=t[6],h=t[7];return t!==e&&(e[8]=t[8],e[9]=t[9],e[10]=t[10],e[11]=t[11],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*i+l*n,e[1]=o*i+u*n,e[2]=a*i+f*n,e[3]=c*i+h*n,e[4]=l*i-s*n,e[5]=u*i-o*n,e[6]=f*i-a*n,e[7]=h*i-c*n,e}function f2(e,t){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=t[0],e[13]=t[1],e[14]=t[2],e[15]=1,e}function h2(e,t){return e[0]=t[0],e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=t[1],e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=t[2],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function p2(e,t,r){var n=r[0],i=r[1],s=r[2],o=Math.hypot(n,i,s),a,c,l;return o<Z?null:(o=1/o,n*=o,i*=o,s*=o,a=Math.sin(t),c=Math.cos(t),l=1-c,e[0]=n*n*l+c,e[1]=i*n*l+s*a,e[2]=s*n*l-i*a,e[3]=0,e[4]=n*i*l-s*a,e[5]=i*i*l+c,e[6]=s*i*l+n*a,e[7]=0,e[8]=n*s*l+i*a,e[9]=i*s*l-n*a,e[10]=s*s*l+c,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e)}function d2(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n,e[6]=r,e[7]=0,e[8]=0,e[9]=-r,e[10]=n,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function m2(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=n,e[1]=0,e[2]=-r,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=r,e[9]=0,e[10]=n,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function g2(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=n,e[1]=r,e[2]=0,e[3]=0,e[4]=-r,e[5]=n,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Am(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=n+n,c=i+i,l=s+s,u=n*a,f=n*c,h=n*l,p=i*c,d=i*l,m=s*l,g=o*a,y=o*c,I=o*l;return e[0]=1-(p+m),e[1]=f+I,e[2]=h-y,e[3]=0,e[4]=f-I,e[5]=1-(u+m),e[6]=d+g,e[7]=0,e[8]=h+y,e[9]=d-g,e[10]=1-(u+p),e[11]=0,e[12]=r[0],e[13]=r[1],e[14]=r[2],e[15]=1,e}function y2(e,t){var r=new Se(3),n=-t[0],i=-t[1],s=-t[2],o=t[3],a=t[4],c=t[5],l=t[6],u=t[7],f=n*n+i*i+s*s+o*o;return f>0?(r[0]=(a*o+u*n+c*s-l*i)*2/f,r[1]=(c*o+u*i+l*n-a*s)*2/f,r[2]=(l*o+u*s+a*i-c*n)*2/f):(r[0]=(a*o+u*n+c*s-l*i)*2,r[1]=(c*o+u*i+l*n-a*s)*2,r[2]=(l*o+u*s+a*i-c*n)*2),Am(e,t,r),e}function v2(e,t){return e[0]=t[12],e[1]=t[13],e[2]=t[14],e}function Mm(e,t){var r=t[0],n=t[1],i=t[2],s=t[4],o=t[5],a=t[6],c=t[8],l=t[9],u=t[10];return e[0]=Math.hypot(r,n,i),e[1]=Math.hypot(s,o,a),e[2]=Math.hypot(c,l,u),e}function x2(e,t){var r=new Se(3);Mm(r,t);var n=1/r[0],i=1/r[1],s=1/r[2],o=t[0]*n,a=t[1]*i,c=t[2]*s,l=t[4]*n,u=t[5]*i,f=t[6]*s,h=t[8]*n,p=t[9]*i,d=t[10]*s,m=o+u+d,g=0;return m>0?(g=Math.sqrt(m+1)*2,e[3]=.25*g,e[0]=(f-p)/g,e[1]=(h-c)/g,e[2]=(a-l)/g):o>u&&o>d?(g=Math.sqrt(1+o-u-d)*2,e[3]=(f-p)/g,e[0]=.25*g,e[1]=(a+l)/g,e[2]=(h+c)/g):u>d?(g=Math.sqrt(1+u-o-d)*2,e[3]=(h-c)/g,e[0]=(a+l)/g,e[1]=.25*g,e[2]=(f+p)/g):(g=Math.sqrt(1+d-o-u)*2,e[3]=(a-l)/g,e[0]=(h+c)/g,e[1]=(f+p)/g,e[2]=.25*g),e}function S2(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=t[3],c=i+i,l=s+s,u=o+o,f=i*c,h=i*l,p=i*u,d=s*l,m=s*u,g=o*u,y=a*c,I=a*l,_=a*u,E=n[0],b=n[1],C=n[2];return e[0]=(1-(d+g))*E,e[1]=(h+_)*E,e[2]=(p-I)*E,e[3]=0,e[4]=(h-_)*b,e[5]=(1-(f+g))*b,e[6]=(m+y)*b,e[7]=0,e[8]=(p+I)*C,e[9]=(m-y)*C,e[10]=(1-(f+d))*C,e[11]=0,e[12]=r[0],e[13]=r[1],e[14]=r[2],e[15]=1,e}function w2(e,t,r,n,i){var s=t[0],o=t[1],a=t[2],c=t[3],l=s+s,u=o+o,f=a+a,h=s*l,p=s*u,d=s*f,m=o*u,g=o*f,y=a*f,I=c*l,_=c*u,E=c*f,b=n[0],C=n[1],v=n[2],w=i[0],x=i[1],k=i[2],M=(1-(m+y))*b,P=(p+E)*b,F=(d-_)*b,S=(p-E)*C,O=(1-(h+y))*C,R=(g+I)*C,N=(d+_)*v,U=(g-I)*v,L=(1-(h+m))*v;return e[0]=M,e[1]=P,e[2]=F,e[3]=0,e[4]=S,e[5]=O,e[6]=R,e[7]=0,e[8]=N,e[9]=U,e[10]=L,e[11]=0,e[12]=r[0]+w-(M*w+S*x+N*k),e[13]=r[1]+x-(P*w+O*x+U*k),e[14]=r[2]+k-(F*w+R*x+L*k),e[15]=1,e}function E2(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r+r,a=n+n,c=i+i,l=r*o,u=n*o,f=n*a,h=i*o,p=i*a,d=i*c,m=s*o,g=s*a,y=s*c;return e[0]=1-f-d,e[1]=u+y,e[2]=h-g,e[3]=0,e[4]=u-y,e[5]=1-l-d,e[6]=p+m,e[7]=0,e[8]=h+g,e[9]=p-m,e[10]=1-l-f,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function b2(e,t,r,n,i,s,o){var a=1/(r-t),c=1/(i-n),l=1/(s-o);return e[0]=s*2*a,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=s*2*c,e[6]=0,e[7]=0,e[8]=(r+t)*a,e[9]=(i+n)*c,e[10]=(o+s)*l,e[11]=-1,e[12]=0,e[13]=0,e[14]=o*s*2*l,e[15]=0,e}function I2(e,t,r,n,i){var s=1/Math.tan(t/2),o;return e[0]=s/r,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=s,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=-1,e[12]=0,e[13]=0,e[15]=0,i!=null&&i!==1/0?(o=1/(n-i),e[10]=(i+n)*o,e[14]=2*i*n*o):(e[10]=-1,e[14]=-2*n),e}function _2(e,t,r,n){var i=Math.tan(t.upDegrees*Math.PI/180),s=Math.tan(t.downDegrees*Math.PI/180),o=Math.tan(t.leftDegrees*Math.PI/180),a=Math.tan(t.rightDegrees*Math.PI/180),c=2/(o+a),l=2/(i+s);return e[0]=c,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=l,e[6]=0,e[7]=0,e[8]=-((o-a)*c*.5),e[9]=(i-s)*l*.5,e[10]=n/(r-n),e[11]=-1,e[12]=0,e[13]=0,e[14]=n*r/(r-n),e[15]=0,e}function C2(e,t,r,n,i,s,o){var a=1/(t-r),c=1/(n-i),l=1/(s-o);return e[0]=-2*a,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=-2*c,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=2*l,e[11]=0,e[12]=(t+r)*a,e[13]=(i+n)*c,e[14]=(o+s)*l,e[15]=1,e}function A2(e,t,r,n){var i,s,o,a,c,l,u,f,h,p,d=t[0],m=t[1],g=t[2],y=n[0],I=n[1],_=n[2],E=r[0],b=r[1],C=r[2];return Math.abs(d-E)<Z&&Math.abs(m-b)<Z&&Math.abs(g-C)<Z?_m(e):(u=d-E,f=m-b,h=g-C,p=1/Math.hypot(u,f,h),u*=p,f*=p,h*=p,i=I*h-_*f,s=_*u-y*h,o=y*f-I*u,p=Math.hypot(i,s,o),p?(p=1/p,i*=p,s*=p,o*=p):(i=0,s=0,o=0),a=f*o-h*s,c=h*i-u*o,l=u*s-f*i,p=Math.hypot(a,c,l),p?(p=1/p,a*=p,c*=p,l*=p):(a=0,c=0,l=0),e[0]=i,e[1]=a,e[2]=u,e[3]=0,e[4]=s,e[5]=c,e[6]=f,e[7]=0,e[8]=o,e[9]=l,e[10]=h,e[11]=0,e[12]=-(i*d+s*m+o*g),e[13]=-(a*d+c*m+l*g),e[14]=-(u*d+f*m+h*g),e[15]=1,e)}function M2(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=n[0],c=n[1],l=n[2],u=i-r[0],f=s-r[1],h=o-r[2],p=u*u+f*f+h*h;p>0&&(p=1/Math.sqrt(p),u*=p,f*=p,h*=p);var d=c*h-l*f,m=l*u-a*h,g=a*f-c*u;return p=d*d+m*m+g*g,p>0&&(p=1/Math.sqrt(p),d*=p,m*=p,g*=p),e[0]=d,e[1]=m,e[2]=g,e[3]=0,e[4]=f*g-h*m,e[5]=h*d-u*g,e[6]=u*m-f*d,e[7]=0,e[8]=u,e[9]=f,e[10]=h,e[11]=0,e[12]=i,e[13]=s,e[14]=o,e[15]=1,e}function k2(e){return"mat4("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+", "+e[4]+", "+e[5]+", "+e[6]+", "+e[7]+", "+e[8]+", "+e[9]+", "+e[10]+", "+e[11]+", "+e[12]+", "+e[13]+", "+e[14]+", "+e[15]+")"}function T2(e){return Math.hypot(e[0],e[1],e[3],e[4],e[5],e[6],e[7],e[8],e[9],e[10],e[11],e[12],e[13],e[14],e[15])}function N2(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e[3]=t[3]+r[3],e[4]=t[4]+r[4],e[5]=t[5]+r[5],e[6]=t[6]+r[6],e[7]=t[7]+r[7],e[8]=t[8]+r[8],e[9]=t[9]+r[9],e[10]=t[10]+r[10],e[11]=t[11]+r[11],e[12]=t[12]+r[12],e[13]=t[13]+r[13],e[14]=t[14]+r[14],e[15]=t[15]+r[15],e}function km(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e[3]=t[3]-r[3],e[4]=t[4]-r[4],e[5]=t[5]-r[5],e[6]=t[6]-r[6],e[7]=t[7]-r[7],e[8]=t[8]-r[8],e[9]=t[9]-r[9],e[10]=t[10]-r[10],e[11]=t[11]-r[11],e[12]=t[12]-r[12],e[13]=t[13]-r[13],e[14]=t[14]-r[14],e[15]=t[15]-r[15],e}function D2(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e[3]=t[3]*r,e[4]=t[4]*r,e[5]=t[5]*r,e[6]=t[6]*r,e[7]=t[7]*r,e[8]=t[8]*r,e[9]=t[9]*r,e[10]=t[10]*r,e[11]=t[11]*r,e[12]=t[12]*r,e[13]=t[13]*r,e[14]=t[14]*r,e[15]=t[15]*r,e}function P2(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e[3]=t[3]+r[3]*n,e[4]=t[4]+r[4]*n,e[5]=t[5]+r[5]*n,e[6]=t[6]+r[6]*n,e[7]=t[7]+r[7]*n,e[8]=t[8]+r[8]*n,e[9]=t[9]+r[9]*n,e[10]=t[10]+r[10]*n,e[11]=t[11]+r[11]*n,e[12]=t[12]+r[12]*n,e[13]=t[13]+r[13]*n,e[14]=t[14]+r[14]*n,e[15]=t[15]+r[15]*n,e}function R2(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]&&e[3]===t[3]&&e[4]===t[4]&&e[5]===t[5]&&e[6]===t[6]&&e[7]===t[7]&&e[8]===t[8]&&e[9]===t[9]&&e[10]===t[10]&&e[11]===t[11]&&e[12]===t[12]&&e[13]===t[13]&&e[14]===t[14]&&e[15]===t[15]}function O2(e,t){var r=e[0],n=e[1],i=e[2],s=e[3],o=e[4],a=e[5],c=e[6],l=e[7],u=e[8],f=e[9],h=e[10],p=e[11],d=e[12],m=e[13],g=e[14],y=e[15],I=t[0],_=t[1],E=t[2],b=t[3],C=t[4],v=t[5],w=t[6],x=t[7],k=t[8],M=t[9],P=t[10],F=t[11],S=t[12],O=t[13],R=t[14],N=t[15];return Math.abs(r-I)<=Z*Math.max(1,Math.abs(r),Math.abs(I))&&Math.abs(n-_)<=Z*Math.max(1,Math.abs(n),Math.abs(_))&&Math.abs(i-E)<=Z*Math.max(1,Math.abs(i),Math.abs(E))&&Math.abs(s-b)<=Z*Math.max(1,Math.abs(s),Math.abs(b))&&Math.abs(o-C)<=Z*Math.max(1,Math.abs(o),Math.abs(C))&&Math.abs(a-v)<=Z*Math.max(1,Math.abs(a),Math.abs(v))&&Math.abs(c-w)<=Z*Math.max(1,Math.abs(c),Math.abs(w))&&Math.abs(l-x)<=Z*Math.max(1,Math.abs(l),Math.abs(x))&&Math.abs(u-k)<=Z*Math.max(1,Math.abs(u),Math.abs(k))&&Math.abs(f-M)<=Z*Math.max(1,Math.abs(f),Math.abs(M))&&Math.abs(h-P)<=Z*Math.max(1,Math.abs(h),Math.abs(P))&&Math.abs(p-F)<=Z*Math.max(1,Math.abs(p),Math.abs(F))&&Math.abs(d-S)<=Z*Math.max(1,Math.abs(d),Math.abs(S))&&Math.abs(m-O)<=Z*Math.max(1,Math.abs(m),Math.abs(O))&&Math.abs(g-R)<=Z*Math.max(1,Math.abs(g),Math.abs(R))&&Math.abs(y-N)<=Z*Math.max(1,Math.abs(y),Math.abs(N))}var U2=Cm,L2=km;var Qt={};Pi(Qt,{add:()=>iC,calculateW:()=>Y_,clone:()=>eC,conjugate:()=>X_,copy:()=>rC,create:()=>Kl,dot:()=>Hm,equals:()=>uC,exactEquals:()=>lC,exp:()=>qm,fromEuler:()=>Z_,fromMat3:()=>Ym,fromValues:()=>tC,getAngle:()=>j_,getAxisAngle:()=>V_,identity:()=>$_,invert:()=>W_,len:()=>aC,length:()=>Wm,lerp:()=>oC,ln:()=>Km,mul:()=>sC,multiply:()=>Gm,normalize:()=>Yl,pow:()=>J_,random:()=>H_,rotateX:()=>G_,rotateY:()=>q_,rotateZ:()=>K_,rotationTo:()=>fC,scale:()=>Jm,set:()=>nC,setAxes:()=>pC,setAxisAngle:()=>jm,slerp:()=>uo,sqlerp:()=>hC,sqrLen:()=>cC,squaredLength:()=>Xm,str:()=>Q_});var B={};Pi(B,{add:()=>$2,angle:()=>a_,bezier:()=>Q2,ceil:()=>V2,clone:()=>F2,copy:()=>B2,create:()=>oo,cross:()=>Gi,dist:()=>m_,distance:()=>Rm,div:()=>d_,divide:()=>Pm,dot:()=>ao,equals:()=>f_,exactEquals:()=>u_,floor:()=>j2,forEach:()=>v_,fromValues:()=>Mn,hermite:()=>Z2,inverse:()=>W2,len:()=>Rl,length:()=>Tm,lerp:()=>X2,max:()=>q2,min:()=>G2,mul:()=>p_,multiply:()=>Dm,negate:()=>H2,normalize:()=>ji,random:()=>e_,rotateX:()=>i_,rotateY:()=>s_,rotateZ:()=>o_,round:()=>K2,scale:()=>Y2,scaleAndAdd:()=>J2,set:()=>z2,sqrDist:()=>g_,sqrLen:()=>y_,squaredDistance:()=>Om,squaredLength:()=>Um,str:()=>l_,sub:()=>h_,subtract:()=>Nm,transformMat3:()=>r_,transformMat4:()=>t_,transformQuat:()=>n_,zero:()=>c_});function oo(){var e=new Se(3);return Se!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e}function F2(e){var t=new Se(3);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t}function Tm(e){var t=e[0],r=e[1],n=e[2];return Math.hypot(t,r,n)}function Mn(e,t,r){var n=new Se(3);return n[0]=e,n[1]=t,n[2]=r,n}function B2(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e}function z2(e,t,r,n){return e[0]=t,e[1]=r,e[2]=n,e}function $2(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e}function Nm(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e}function Dm(e,t,r){return e[0]=t[0]*r[0],e[1]=t[1]*r[1],e[2]=t[2]*r[2],e}function Pm(e,t,r){return e[0]=t[0]/r[0],e[1]=t[1]/r[1],e[2]=t[2]/r[2],e}function V2(e,t){return e[0]=Math.ceil(t[0]),e[1]=Math.ceil(t[1]),e[2]=Math.ceil(t[2]),e}function j2(e,t){return e[0]=Math.floor(t[0]),e[1]=Math.floor(t[1]),e[2]=Math.floor(t[2]),e}function G2(e,t,r){return e[0]=Math.min(t[0],r[0]),e[1]=Math.min(t[1],r[1]),e[2]=Math.min(t[2],r[2]),e}function q2(e,t,r){return e[0]=Math.max(t[0],r[0]),e[1]=Math.max(t[1],r[1]),e[2]=Math.max(t[2],r[2]),e}function K2(e,t){return e[0]=Math.round(t[0]),e[1]=Math.round(t[1]),e[2]=Math.round(t[2]),e}function Y2(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e}function J2(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e}function Rm(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2];return Math.hypot(r,n,i)}function Om(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2];return r*r+n*n+i*i}function Um(e){var t=e[0],r=e[1],n=e[2];return t*t+r*r+n*n}function H2(e,t){return e[0]=-t[0],e[1]=-t[1],e[2]=-t[2],e}function W2(e,t){return e[0]=1/t[0],e[1]=1/t[1],e[2]=1/t[2],e}function ji(e,t){var r=t[0],n=t[1],i=t[2],s=r*r+n*n+i*i;return s>0&&(s=1/Math.sqrt(s)),e[0]=t[0]*s,e[1]=t[1]*s,e[2]=t[2]*s,e}function ao(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]}function Gi(e,t,r){var n=t[0],i=t[1],s=t[2],o=r[0],a=r[1],c=r[2];return e[0]=i*c-s*a,e[1]=s*o-n*c,e[2]=n*a-i*o,e}function X2(e,t,r,n){var i=t[0],s=t[1],o=t[2];return e[0]=i+n*(r[0]-i),e[1]=s+n*(r[1]-s),e[2]=o+n*(r[2]-o),e}function Z2(e,t,r,n,i,s){var o=s*s,a=o*(2*s-3)+1,c=o*(s-2)+s,l=o*(s-1),u=o*(3-2*s);return e[0]=t[0]*a+r[0]*c+n[0]*l+i[0]*u,e[1]=t[1]*a+r[1]*c+n[1]*l+i[1]*u,e[2]=t[2]*a+r[2]*c+n[2]*l+i[2]*u,e}function Q2(e,t,r,n,i,s){var o=1-s,a=o*o,c=s*s,l=a*o,u=3*s*a,f=3*c*o,h=c*s;return e[0]=t[0]*l+r[0]*u+n[0]*f+i[0]*h,e[1]=t[1]*l+r[1]*u+n[1]*f+i[1]*h,e[2]=t[2]*l+r[2]*u+n[2]*f+i[2]*h,e}function e_(e,t){t=t||1;var r=Mt()*2*Math.PI,n=Mt()*2-1,i=Math.sqrt(1-n*n)*t;return e[0]=Math.cos(r)*i,e[1]=Math.sin(r)*i,e[2]=n*t,e}function t_(e,t,r){var n=t[0],i=t[1],s=t[2],o=r[3]*n+r[7]*i+r[11]*s+r[15];return o=o||1,e[0]=(r[0]*n+r[4]*i+r[8]*s+r[12])/o,e[1]=(r[1]*n+r[5]*i+r[9]*s+r[13])/o,e[2]=(r[2]*n+r[6]*i+r[10]*s+r[14])/o,e}function r_(e,t,r){var n=t[0],i=t[1],s=t[2];return e[0]=n*r[0]+i*r[3]+s*r[6],e[1]=n*r[1]+i*r[4]+s*r[7],e[2]=n*r[2]+i*r[5]+s*r[8],e}function n_(e,t,r){var n=r[0],i=r[1],s=r[2],o=r[3],a=t[0],c=t[1],l=t[2],u=i*l-s*c,f=s*a-n*l,h=n*c-i*a,p=i*h-s*f,d=s*u-n*h,m=n*f-i*u,g=o*2;return u*=g,f*=g,h*=g,p*=2,d*=2,m*=2,e[0]=a+u+p,e[1]=c+f+d,e[2]=l+h+m,e}function i_(e,t,r,n){var i=[],s=[];return i[0]=t[0]-r[0],i[1]=t[1]-r[1],i[2]=t[2]-r[2],s[0]=i[0],s[1]=i[1]*Math.cos(n)-i[2]*Math.sin(n),s[2]=i[1]*Math.sin(n)+i[2]*Math.cos(n),e[0]=s[0]+r[0],e[1]=s[1]+r[1],e[2]=s[2]+r[2],e}function s_(e,t,r,n){var i=[],s=[];return i[0]=t[0]-r[0],i[1]=t[1]-r[1],i[2]=t[2]-r[2],s[0]=i[2]*Math.sin(n)+i[0]*Math.cos(n),s[1]=i[1],s[2]=i[2]*Math.cos(n)-i[0]*Math.sin(n),e[0]=s[0]+r[0],e[1]=s[1]+r[1],e[2]=s[2]+r[2],e}function o_(e,t,r,n){var i=[],s=[];return i[0]=t[0]-r[0],i[1]=t[1]-r[1],i[2]=t[2]-r[2],s[0]=i[0]*Math.cos(n)-i[1]*Math.sin(n),s[1]=i[0]*Math.sin(n)+i[1]*Math.cos(n),s[2]=i[2],e[0]=s[0]+r[0],e[1]=s[1]+r[1],e[2]=s[2]+r[2],e}function a_(e,t){var r=Mn(e[0],e[1],e[2]),n=Mn(t[0],t[1],t[2]);ji(r,r),ji(n,n);var i=ao(r,n);return i>1?0:i<-1?Math.PI:Math.acos(i)}function c_(e){return e[0]=0,e[1]=0,e[2]=0,e}function l_(e){return"vec3("+e[0]+", "+e[1]+", "+e[2]+")"}function u_(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]}function f_(e,t){var r=e[0],n=e[1],i=e[2],s=t[0],o=t[1],a=t[2];return Math.abs(r-s)<=Z*Math.max(1,Math.abs(r),Math.abs(s))&&Math.abs(n-o)<=Z*Math.max(1,Math.abs(n),Math.abs(o))&&Math.abs(i-a)<=Z*Math.max(1,Math.abs(i),Math.abs(a))}var h_=Nm,p_=Dm,d_=Pm,m_=Rm,g_=Om,Rl=Tm,y_=Um,v_=function(){var e=oo();return function(t,r,n,i,s,o){var a,c;for(r||(r=3),n||(n=0),i?c=Math.min(i*r+n,t.length):c=t.length,a=n;a<c;a+=r)e[0]=t[a],e[1]=t[a+1],e[2]=t[a+2],s(e,e,o),t[a]=e[0],t[a+1]=e[1],t[a+2]=e[2];return t}}();var gr={};Pi(gr,{add:()=>Bl,ceil:()=>x_,clone:()=>Ol,copy:()=>Ll,create:()=>Lm,cross:()=>A_,dist:()=>U_,distance:()=>$m,div:()=>O_,divide:()=>zm,dot:()=>Vl,equals:()=>ql,exactEquals:()=>Gl,floor:()=>S_,forEach:()=>z_,fromValues:()=>Ul,inverse:()=>C_,len:()=>F_,length:()=>co,lerp:()=>jl,max:()=>E_,min:()=>w_,mul:()=>R_,multiply:()=>Bm,negate:()=>__,normalize:()=>$l,random:()=>M_,round:()=>b_,scale:()=>zl,scaleAndAdd:()=>I_,set:()=>Fl,sqrDist:()=>L_,sqrLen:()=>B_,squaredDistance:()=>Vm,squaredLength:()=>lo,str:()=>D_,sub:()=>P_,subtract:()=>Fm,transformMat4:()=>k_,transformQuat:()=>T_,zero:()=>N_});function Lm(){var e=new Se(4);return Se!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0,e[3]=0),e}function Ol(e){var t=new Se(4);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t}function Ul(e,t,r,n){var i=new Se(4);return i[0]=e,i[1]=t,i[2]=r,i[3]=n,i}function Ll(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e}function Fl(e,t,r,n,i){return e[0]=t,e[1]=r,e[2]=n,e[3]=i,e}function Bl(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e[3]=t[3]+r[3],e}function Fm(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e[3]=t[3]-r[3],e}function Bm(e,t,r){return e[0]=t[0]*r[0],e[1]=t[1]*r[1],e[2]=t[2]*r[2],e[3]=t[3]*r[3],e}function zm(e,t,r){return e[0]=t[0]/r[0],e[1]=t[1]/r[1],e[2]=t[2]/r[2],e[3]=t[3]/r[3],e}function x_(e,t){return e[0]=Math.ceil(t[0]),e[1]=Math.ceil(t[1]),e[2]=Math.ceil(t[2]),e[3]=Math.ceil(t[3]),e}function S_(e,t){return e[0]=Math.floor(t[0]),e[1]=Math.floor(t[1]),e[2]=Math.floor(t[2]),e[3]=Math.floor(t[3]),e}function w_(e,t,r){return e[0]=Math.min(t[0],r[0]),e[1]=Math.min(t[1],r[1]),e[2]=Math.min(t[2],r[2]),e[3]=Math.min(t[3],r[3]),e}function E_(e,t,r){return e[0]=Math.max(t[0],r[0]),e[1]=Math.max(t[1],r[1]),e[2]=Math.max(t[2],r[2]),e[3]=Math.max(t[3],r[3]),e}function b_(e,t){return e[0]=Math.round(t[0]),e[1]=Math.round(t[1]),e[2]=Math.round(t[2]),e[3]=Math.round(t[3]),e}function zl(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e[3]=t[3]*r,e}function I_(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e[3]=t[3]+r[3]*n,e}function $m(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2],s=t[3]-e[3];return Math.hypot(r,n,i,s)}function Vm(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2],s=t[3]-e[3];return r*r+n*n+i*i+s*s}function co(e){var t=e[0],r=e[1],n=e[2],i=e[3];return Math.hypot(t,r,n,i)}function lo(e){var t=e[0],r=e[1],n=e[2],i=e[3];return t*t+r*r+n*n+i*i}function __(e,t){return e[0]=-t[0],e[1]=-t[1],e[2]=-t[2],e[3]=-t[3],e}function C_(e,t){return e[0]=1/t[0],e[1]=1/t[1],e[2]=1/t[2],e[3]=1/t[3],e}function $l(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r*r+n*n+i*i+s*s;return o>0&&(o=1/Math.sqrt(o)),e[0]=r*o,e[1]=n*o,e[2]=i*o,e[3]=s*o,e}function Vl(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]+e[3]*t[3]}function A_(e,t,r,n){var i=r[0]*n[1]-r[1]*n[0],s=r[0]*n[2]-r[2]*n[0],o=r[0]*n[3]-r[3]*n[0],a=r[1]*n[2]-r[2]*n[1],c=r[1]*n[3]-r[3]*n[1],l=r[2]*n[3]-r[3]*n[2],u=t[0],f=t[1],h=t[2],p=t[3];return e[0]=f*l-h*c+p*a,e[1]=-(u*l)+h*o-p*s,e[2]=u*c-f*o+p*i,e[3]=-(u*a)+f*s-h*i,e}function jl(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=t[3];return e[0]=i+n*(r[0]-i),e[1]=s+n*(r[1]-s),e[2]=o+n*(r[2]-o),e[3]=a+n*(r[3]-a),e}function M_(e,t){t=t||1;var r,n,i,s,o,a;do r=Mt()*2-1,n=Mt()*2-1,o=r*r+n*n;while(o>=1);do i=Mt()*2-1,s=Mt()*2-1,a=i*i+s*s;while(a>=1);var c=Math.sqrt((1-o)/a);return e[0]=t*r,e[1]=t*n,e[2]=t*i*c,e[3]=t*s*c,e}function k_(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3];return e[0]=r[0]*n+r[4]*i+r[8]*s+r[12]*o,e[1]=r[1]*n+r[5]*i+r[9]*s+r[13]*o,e[2]=r[2]*n+r[6]*i+r[10]*s+r[14]*o,e[3]=r[3]*n+r[7]*i+r[11]*s+r[15]*o,e}function T_(e,t,r){var n=t[0],i=t[1],s=t[2],o=r[0],a=r[1],c=r[2],l=r[3],u=l*n+a*s-c*i,f=l*i+c*n-o*s,h=l*s+o*i-a*n,p=-o*n-a*i-c*s;return e[0]=u*l+p*-o+f*-c-h*-a,e[1]=f*l+p*-a+h*-o-u*-c,e[2]=h*l+p*-c+u*-a-f*-o,e[3]=t[3],e}function N_(e){return e[0]=0,e[1]=0,e[2]=0,e[3]=0,e}function D_(e){return"vec4("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+")"}function Gl(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]&&e[3]===t[3]}function ql(e,t){var r=e[0],n=e[1],i=e[2],s=e[3],o=t[0],a=t[1],c=t[2],l=t[3];return Math.abs(r-o)<=Z*Math.max(1,Math.abs(r),Math.abs(o))&&Math.abs(n-a)<=Z*Math.max(1,Math.abs(n),Math.abs(a))&&Math.abs(i-c)<=Z*Math.max(1,Math.abs(i),Math.abs(c))&&Math.abs(s-l)<=Z*Math.max(1,Math.abs(s),Math.abs(l))}var P_=Fm,R_=Bm,O_=zm,U_=$m,L_=Vm,F_=co,B_=lo,z_=function(){var e=Lm();return function(t,r,n,i,s,o){var a,c;for(r||(r=4),n||(n=0),i?c=Math.min(i*r+n,t.length):c=t.length,a=n;a<c;a+=r)e[0]=t[a],e[1]=t[a+1],e[2]=t[a+2],e[3]=t[a+3],s(e,e,o),t[a]=e[0],t[a+1]=e[1],t[a+2]=e[2],t[a+3]=e[3];return t}}();function Kl(){var e=new Se(4);return Se!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e[3]=1,e}function $_(e){return e[0]=0,e[1]=0,e[2]=0,e[3]=1,e}function jm(e,t,r){r=r*.5;var n=Math.sin(r);return e[0]=n*t[0],e[1]=n*t[1],e[2]=n*t[2],e[3]=Math.cos(r),e}function V_(e,t){var r=Math.acos(t[3])*2,n=Math.sin(r/2);return n>Z?(e[0]=t[0]/n,e[1]=t[1]/n,e[2]=t[2]/n):(e[0]=1,e[1]=0,e[2]=0),r}function j_(e,t){var r=Hm(e,t);return Math.acos(2*r*r-1)}function Gm(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=r[0],c=r[1],l=r[2],u=r[3];return e[0]=n*u+o*a+i*l-s*c,e[1]=i*u+o*c+s*a-n*l,e[2]=s*u+o*l+n*c-i*a,e[3]=o*u-n*a-i*c-s*l,e}function G_(e,t,r){r*=.5;var n=t[0],i=t[1],s=t[2],o=t[3],a=Math.sin(r),c=Math.cos(r);return e[0]=n*c+o*a,e[1]=i*c+s*a,e[2]=s*c-i*a,e[3]=o*c-n*a,e}function q_(e,t,r){r*=.5;var n=t[0],i=t[1],s=t[2],o=t[3],a=Math.sin(r),c=Math.cos(r);return e[0]=n*c-s*a,e[1]=i*c+o*a,e[2]=s*c+n*a,e[3]=o*c-i*a,e}function K_(e,t,r){r*=.5;var n=t[0],i=t[1],s=t[2],o=t[3],a=Math.sin(r),c=Math.cos(r);return e[0]=n*c+i*a,e[1]=i*c-n*a,e[2]=s*c+o*a,e[3]=o*c-s*a,e}function Y_(e,t){var r=t[0],n=t[1],i=t[2];return e[0]=r,e[1]=n,e[2]=i,e[3]=Math.sqrt(Math.abs(1-r*r-n*n-i*i)),e}function qm(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=Math.sqrt(r*r+n*n+i*i),a=Math.exp(s),c=o>0?a*Math.sin(o)/o:0;return e[0]=r*c,e[1]=n*c,e[2]=i*c,e[3]=a*Math.cos(o),e}function Km(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=Math.sqrt(r*r+n*n+i*i),a=o>0?Math.atan2(o,s)/o:0;return e[0]=r*a,e[1]=n*a,e[2]=i*a,e[3]=.5*Math.log(r*r+n*n+i*i+s*s),e}function J_(e,t,r){return Km(e,t),Jm(e,e,r),qm(e,e),e}function uo(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=t[3],c=r[0],l=r[1],u=r[2],f=r[3],h,p,d,m,g;return p=i*c+s*l+o*u+a*f,p<0&&(p=-p,c=-c,l=-l,u=-u,f=-f),1-p>Z?(h=Math.acos(p),d=Math.sin(h),m=Math.sin((1-n)*h)/d,g=Math.sin(n*h)/d):(m=1-n,g=n),e[0]=m*i+g*c,e[1]=m*s+g*l,e[2]=m*o+g*u,e[3]=m*a+g*f,e}function H_(e){var t=Mt(),r=Mt(),n=Mt(),i=Math.sqrt(1-t),s=Math.sqrt(t);return e[0]=i*Math.sin(2*Math.PI*r),e[1]=i*Math.cos(2*Math.PI*r),e[2]=s*Math.sin(2*Math.PI*n),e[3]=s*Math.cos(2*Math.PI*n),e}function W_(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r*r+n*n+i*i+s*s,a=o?1/o:0;return e[0]=-r*a,e[1]=-n*a,e[2]=-i*a,e[3]=s*a,e}function X_(e,t){return e[0]=-t[0],e[1]=-t[1],e[2]=-t[2],e[3]=t[3],e}function Ym(e,t){var r=t[0]+t[4]+t[8],n;if(r>0)n=Math.sqrt(r+1),e[3]=.5*n,n=.5/n,e[0]=(t[5]-t[7])*n,e[1]=(t[6]-t[2])*n,e[2]=(t[1]-t[3])*n;else{var i=0;t[4]>t[0]&&(i=1),t[8]>t[i*3+i]&&(i=2);var s=(i+1)%3,o=(i+2)%3;n=Math.sqrt(t[i*3+i]-t[s*3+s]-t[o*3+o]+1),e[i]=.5*n,n=.5/n,e[3]=(t[s*3+o]-t[o*3+s])*n,e[s]=(t[s*3+i]+t[i*3+s])*n,e[o]=(t[o*3+i]+t[i*3+o])*n}return e}function Z_(e,t,r,n){var i=.5*Math.PI/180;t*=i,r*=i,n*=i;var s=Math.sin(t),o=Math.cos(t),a=Math.sin(r),c=Math.cos(r),l=Math.sin(n),u=Math.cos(n);return e[0]=s*c*u-o*a*l,e[1]=o*a*u+s*c*l,e[2]=o*c*l-s*a*u,e[3]=o*c*u+s*a*l,e}function Q_(e){return"quat("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+")"}var eC=Ol,tC=Ul,rC=Ll,nC=Fl,iC=Bl,sC=Gm,Jm=zl,Hm=Vl,oC=jl,Wm=co,aC=Wm,Xm=lo,cC=Xm,Yl=$l,lC=Gl,uC=ql,fC=function(){var e=oo(),t=Mn(1,0,0),r=Mn(0,1,0);return function(n,i,s){var o=ao(i,s);return o<-.999999?(Gi(e,t,i),Rl(e)<1e-6&&Gi(e,r,i),ji(e,e),jm(n,e,Math.PI),n):o>.999999?(n[0]=0,n[1]=0,n[2]=0,n[3]=1,n):(Gi(e,i,s),n[0]=e[0],n[1]=e[1],n[2]=e[2],n[3]=1+o,Yl(n,n))}}(),hC=function(){var e=Kl(),t=Kl();return function(r,n,i,s,o,a){return uo(e,n,o,a),uo(t,i,s,a),uo(r,e,t,2*a*(1-a)),r}}(),pC=function(){var e=Pl();return function(t,r,n,i){return e[0]=n[0],e[3]=n[1],e[6]=n[2],e[1]=i[0],e[4]=i[1],e[7]=i[2],e[2]=-r[0],e[5]=-r[1],e[8]=-r[2],Yl(t,Ym(t,e))}}();function Zm(e,t){let r=e.length,n=0;for(let i=0;i<r;++i)t(e[i],i,e)&&(e[n]=e[i],++n);e.length=n}function Qm(e,t,r){let n=new e.constructor(e.length);for(let i=0;i<t*r;i+=r)for(let s=0;s<r;s++){let o=i/r;n[s*t+o]=e[i+s]}return n}function er(e,t,r,n=0,i=e.length){for(;n<i;){let s=n+i-1>>1,o=r(t,e[s]);if(o>0)n=s+1;else if(o<0)i=s;else return s}return~n}function we(e,t,r){let n=t-e;for(;n>0;){let i=Math.floor(n/2),s=e+i;r(s)?n=i:(e=s+1,n-=i+1)}return e}function qt(e,t){let r=e.length;if(t.length!==r)return!1;for(let n=0;n<r;++n)if(e[n]!==t[n])return!1;return!0}var MR=ge.create();var kR=[B.fromValues(1,0,0),B.fromValues(0,1,0),B.fromValues(0,0,1)],qi=B.fromValues(0,0,0),TR=gr.fromValues(0,0,0,0),eg=B.fromValues(1,1,1),tg=B.fromValues(1/0,1/0,1/0),NR=Qt.create();function Ki(e){return e[0]*e[1]*e[2]}function tr(e){return`${e[0]},${e[1]},${e[2]}`}function rg(e,t,r){let n=t[0],i=t[1],s=t[2];return e[0]=r[0]*n+r[4]*i+r[8]*s,e[1]=r[1]*n+r[5]*i+r[9]*s,e[2]=r[2]*n+r[6]*i+r[10]*s,e}function ng(e,t,r){let n=t[0],i=t[1],s=t[2];return e[0]=r[0]*n+r[1]*i+r[2]*s,e[1]=r[4]*n+r[5]*i+r[6]*s,e[2]=r[8]*n+r[9]*i+r[10]*s,e}function ig(e,t,r,n,i){let s=e;return e[0]=n[0],e[1]=n[1],e[2]=n[2]*i,ge.fromRotationTranslationScale(e,r,t,s)}function fo(e,t){let r=t[0],n=t[1],i=t[2],s=t[4],o=t[5],a=t[6],c=t[8],l=t[9],u=t[10];return e[0]=r,e[1]=n,e[2]=i,e[3]=s,e[4]=o,e[5]=a,e[6]=c,e[7]=l,e[8]=u,e}function ho(e,t){let r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],l=t[7],u=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15];e[0]=s+r,e[1]=l+o,e[2]=p+u,e[3]=y+d,e[4]=s-r,e[5]=l-o,e[6]=p-u,e[7]=y-d,e[8]=s+n,e[9]=l+a,e[10]=p+f,e[11]=y+m,e[12]=s-n,e[13]=l-a,e[14]=p-f,e[15]=y-m;let I=s+i,_=l+c,E=p+h,b=y+g,C=s-i,v=l-c,w=p-h,x=y-g,k=Math.sqrt(I**2+_**2+E**2);e[16]=I/k,e[17]=_/k,e[18]=E/k,e[19]=b/k;let M=Math.sqrt(C**2+v**2+w**2);return e[20]=C/M,e[21]=v/M,e[22]=w/M,e[23]=x/M,e}function po(e,t,r,n,i,s,o){for(let a=0;a<6;++a){let c=o[a*4],l=o[a*4+1],u=o[a*4+2],f=o[a*4+3];if(Math.max(c*e,c*n)+Math.max(l*t,l*i)+Math.max(u*r,u*s)+f<0)return!1}return!0}function sg(e,t,r,n,i,s,o){for(let a=0;a<4;++a){let c=o[a*4],l=o[a*4+1],u=o[a*4+2],f=o[a*4+3];if(Math.max(c*e,c*n)+Math.max(l*t,l*i)+Math.max(u*r,u*s)+f<0)return!1}{let c=o[20],l=o[5*4+1],u=o[5*4+2],f=o[5*4+3],h=Math.max(c*e,c*n)+Math.max(l*t,l*i)+Math.max(u*r,u*s),p=Math.min(c*e,c*n)+Math.min(l*t,l*i)+Math.min(u*r,u*s),d=Math.abs(f)*1e-6;if(p>-f+d||h<-f-d)return!1}return!0}function og(e){if(e[15]===1){let o=2/Math.abs(e[10]),a=2/Math.abs(e[0]),c=2/Math.abs(e[5]);return a*c*o}let t=e[10],n=2*e[14]/(2*t-2),i=(t-1)*n/(t+1);return 4/(e[0]*e[5])/3*(Math.abs(i)**3-Math.abs(n)**3)}function mo(e){if(e[15]===1)return 2/Math.abs(e[10]);let t=e[10],n=2*e[14]/(2*t-2),i=(t-1)*n/(t+1);return Math.abs(i-n)}var DR=B.create();function go(e){let t=typeof e;if(t==="number"||t==="string"){let r=parseFloat(""+e);if(!Number.isNaN(r))return r}throw new Error(`Expected floating-point number, but received: ${JSON.stringify(e)}.`)}function rr(e){let t=go(e);if(Number.isFinite(t))return t;throw new Error(`Expected finite floating-point number, but received: ${t}.`)}function ag(e){let t=go(e);if(Number.isFinite(t)&&t>=0)return t;throw new Error(`Expected finite non-negative floating-point number, but received: ${t}.`)}function Kt(e){if(typeof e=="object"){if(e===null)return"null";if(Array.isArray(e)){let s="[",o=e.length,a=0;if(a<o)for(s+=Kt(e[a]);++a<o;)s+=",",s+=Kt(e[a]);return s+="]",s}let t="{",r=Object.keys(e).sort(),n=0,i=r.length;if(n<i){let s=r[n];for(t+=JSON.stringify(s),t+=":",t+=Kt(e[s]);++n<i;)t+=",",s=r[n],t+=JSON.stringify(s),t+=":",t+=Kt(e[s])}return t+="}",t}return typeof e=="bigint"?e.toString():JSON.stringify(e)}var cg=/('(?:[^'\\]|(?:\\.))*')/,lg=/("(?:[^"\\]|(?:\\.))*")/,dC=new RegExp(`${cg.source}|${lg.source}`),LR=new RegExp(`${lg.source}|${cg.source}`),mC=/^((?:[^"'\\]|(?:\\[^']))*)("|\\')/;function gC(e,t,r,n){if(e.length>=2&&e.charAt(0)===t&&e.charAt(e.length-1)===t){let i=e.substr(1,e.length-2),s=r;for(;i.length>0;){let o=i.match(n);if(o===null){s+=i;break}s+=o[1],o[2]===r?(s+="\\",s+=r):s+=t,i=i.substr(o.index+o[0].length)}return s+=r,s}return e}function yC(e){return gC(e,"'",'"',mC)}function vC(e){let t="";for(;e.length>0;){let r=e.match(dC),n,i;if(r===null)n=e,e="",i="";else{n=e.substr(0,r.index),e=e.substr(r.index+r[0].length);let s=r[1];s!==void 0?i=yC(s):i=r[2]}t+=n.replace(/\(/g,"[").replace(/\)/g,"]").replace("True","true").replace("False","false").replace(/,\s*([}\]])/g,"$1"),t+=i}return t}function ug(e){return JSON.parse(vC(e))}function ct(e,t){if(!Array.isArray(e))throw new Error(`Expected array, but received: ${JSON.stringify(e)}.`);return e.map(t)}function We(e,t,r){let n=e.length;if(!Array.isArray(t)||t.length!==n)throw new Error(`Expected length ${n} array, but received: ${JSON.stringify(t)}.`);for(let i=0;i<n;++i)e[i]=r(t[i],i);return e}function Ee(e){if(typeof e!="object"||e==null||Array.isArray(e))throw new Error(`Expected JSON object, but received: ${JSON.stringify(e)}.`);return e}function Tr(e){let t=parseInt(e,10);if(!Number.isInteger(t))throw new Error(`Expected integer, but received: ${JSON.stringify(e)}.`);return t}function et(e){if(typeof e!="string")throw new Error(`Expected string, but received: ${JSON.stringify(e)}.`);return e}function Jl(e){if(e!==void 0)return et(e)}function re(e,t,r){let n=Object.prototype.hasOwnProperty.call(e,t)?e[t]:void 0;try{return r(n)}catch(i){throw new Error(`Error parsing ${JSON.stringify(t)} property: ${i.message}`)}}function nr(e,t,r,n){return re(e,t,i=>i===void 0?n:r(i))}function kn(e,t,r=/^[a-zA-Z]/){if(typeof e=="string"&&e.match(r)!==null){let n=e.toUpperCase();if(Object.prototype.hasOwnProperty.call(t,n))return t[n]}throw new Error(`Invalid enum value: ${JSON.stringify(e)}.`)}function Yt(e){if(!Array.isArray(e))throw new Error(`Expected array, received: ${JSON.stringify(e)}.`);for(let t of e)if(typeof t!="string")throw new Error(`Expected string, received: ${JSON.stringify(t)}.`);return e}function yr(e){let t;switch(typeof e){case"string":if(e.match(/^(?:0|[1-9][0-9]*)$/)===null)throw new Error(`Expected base-10 number, but received: ${JSON.stringify(e)}`);t=BigInt(e);break;case"number":t=BigInt(e);break;case"bigint":t=e;break;default:throw new Error(`Expected uint64 value, but received: ${JSON.stringify(e)}`)}if(t<0n||t>Dl)throw new Error(`Expected uint64 value, but received: ${t}`);return t}var yo=class{map=new Map;get(t,r){let{map:n}=this,i=n.get(t);return i===void 0?(i=r(),i.registerDisposer(()=>{n.delete(t)}),n.set(t,i)):i.addRef(),i}},Tn=class extends yo{get(t,r){return typeof t!="string"&&(t=Kt(t)),super.get(t,r)}getUncounted(t,r){return this.get(t,()=>new Xs(r())).value}getAsync(t,r,n){return this.getUncounted(t,()=>Yi(n))(r)}};function Yi(e){let t,r,n,i=!1;return async s=>{if(i)return n;let{signal:o}=s;if(o?.throwIfAborted(),n===void 0||r.signal.aborted){t=new eo,r=new Zs;let c=r;n=(async()=>{try{return await e({signal:c.signal,progressListener:t})}catch(l){throw c.signal.aborted&&(n=void 0),l}finally{n!==void 0&&(i=!0),t=void 0,c[Symbol.dispose](),r===c&&(r=void 0)}})()}r.addConsumer(o);let a=t;a.addListener(s.progressListener);try{return await Qs(n,o)}finally{a.removeListener(s.progressListener)}}}function vo(e){let{child:t,next:r,prev:n,compare:i}=e;function s(f){let h=f[t];if(h===null)return null;let p=null;for(;;){let m=h[r],g,y;if(m===null?(g=null,y=h):(g=m[r],y=o(h,m)),y[r]=p,p=y,g===null)break;h=g}let d=p;for(p=p[r];p!==null;){let m=p[r];d=o(d,p),p=m}return d[n]=null,d[r]=null,d}function o(f,h){if(h===null)return f;if(f===null)return h;if(i(h,f)){let d=f;f=h,h=d}let p=f[t];return h[r]=p,h[n]=f,p!==null&&(p[n]=h),f[t]=h,f}function a(f){let h=s(f);return f[r]=null,f[n]=null,f[t]=null,h}function c(f,h){if(f===h)return a(f);let p=h[n],d=h[r];p[t]===h?p[t]=d:p[r]=d,d!==null&&(d[n]=p);let m=o(f,s(h));return h[r]=null,h[n]=null,h[t]=null,m}function*l(f){if(f!==null){let h=f[t];for(yield f;h!==null;){let p=h[r];yield*l(h),h=p}}}function*u(f){if(f!==null){let h=f[t];for(f[t]=null,f[r]=null,f[n]=null,yield f;h!==null;){let p=h[r];h[t]=null,h[r]=null,h[n]=null,yield*l(h),h=p}}}return{compare:i,meld:o,removeMin:a,remove:c,entries:l,removedEntries:u}}var xC=Object.defineProperty,SC=Object.getOwnPropertyDescriptor,Xl=(e,t,r,n)=>{for(var i=n>1?void 0:n?SC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&xC(t,r,i),i};var Ji=!1,wC=0;function mg(){return++wC}var fe=class{child0=null;next0=null;prev0=null;child1=null;next1=null;prev1=null;source=null;key=null;state_=z.NEW;error=null;markGeneration=-1;priority=0;newPriority=0;priorityTier=se.RECENT;newPriorityTier=se.RECENT;systemMemoryBytes_=0;gpuMemoryBytes_=0;downloadSlots_=1;isComputational=!1;requestedState=z.NEW;newRequestedState=z.NEW;downloadAbortController=void 0;initialize(t){this.key=t,this.priority=Number.NEGATIVE_INFINITY,this.priorityTier=se.RECENT,this.newPriority=Number.NEGATIVE_INFINITY,this.newPriorityTier=se.RECENT,this.error=null,this.state=z.NEW,this.requestedState=z.NEW,this.newRequestedState=z.NEW}updatePriorityProperties(){this.priorityTier=this.newPriorityTier,this.priority=this.newPriority,this.newPriorityTier=se.RECENT,this.newPriority=Number.NEGATIVE_INFINITY,this.requestedState=this.newRequestedState,this.newRequestedState=z.NEW}dispose(){this.source=null,this.error=null}get chunkManager(){return this.source.chunkManager}get queueManager(){return this.source.chunkManager.queueManager}downloadFailed(t){this.error=t,this.queueManager.updateChunkState(this,z.FAILED)}downloadSucceeded(){this.requestedState===z.SYSTEM_MEMORY?(this.queueManager.moveChunkToFrontend(this),this.queueManager.updateChunkState(this,z.SYSTEM_MEMORY)):this.queueManager.updateChunkState(this,z.SYSTEM_MEMORY_WORKER)}freeSystemMemory(){}serialize(t,r){t.id=this.key,t.source=this.source.rpcId,t.new=!0}toString(){return this.key}set state(t){if(t===this.state_)return;let r=this.state_;this.state_=t,this.source.chunkStateChanged(this,r)}get state(){return this.state_}set systemMemoryBytes(t){vr(this,-1),this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!1),this.systemMemoryBytes_=t,this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!0),vr(this,1),this.chunkManager.queueManager.scheduleUpdate()}get systemMemoryBytes(){return this.systemMemoryBytes_}set gpuMemoryBytes(t){vr(this,-1),this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!1),this.gpuMemoryBytes_=t,this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!0),vr(this,1),this.chunkManager.queueManager.scheduleUpdate()}get gpuMemoryBytes(){return this.gpuMemoryBytes_}get downloadSlots(){return this.downloadSlots_}set downloadSlots(t){t!==this.downloadSlots_&&(vr(this,-1),this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!1),this.downloadSlots_=t,this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!0),vr(this,1),this.chunkManager.queueManager.scheduleUpdate())}registerListener(t){return this.source?this.source.registerChunkListener(this.key,t):!1}unregisterListener(t){return this.source?this.source.unregisterChunkListener(this.key,t):!1}static priorityLess(t,r){return t.priority<r.priority}static priorityGreater(t,r){return t.priority>r.priority}},EC=2,Hi=class extends rn{constructor(t){super(),this.chunkManager=t,t.queueManager.sources.add(this)}listeners_=new Map;chunks=new Map;freeChunks=new Array;statistics=new Float64Array(mm);sourceQueueLevel=0;disposed(){this.chunkManager.queueManager.sources.delete(this),super.disposed()}getNewChunk_(t){let r=this.freeChunks,n=r.length;if(n>0){let s=r[n-1];return r.length=n-1,s.source=this,s}let i=new t;return i.source=this,i}addChunk(t){let{chunks:r}=this;r.size===0&&this.addRef(),r.set(t.key,t),vr(t,1)}removeChunk(t){let{chunks:r,freeChunks:n}=this;r.delete(t.key),t.dispose(),n[n.length]=t,r.size===0&&this.dispose()}registerChunkListener(t,r){return this.listeners_.has(t)?this.listeners_.get(t).push(r):this.listeners_.set(t,[r]),!0}unregisterChunkListener(t,r){if(!this.listeners_.has(t))return!1;let n=this.listeners_.get(t),i=n.indexOf(r);return i<0?!1:(n.splice(i,1),n.length===0&&this.listeners_.delete(t),!0)}chunkStateChanged(t,r){let{key:n}=t;if(n===null)return;let i=this.listeners_.get(n);if(i!==void 0)for(let s of i.slice())s(t,r)}};function vr(e,t){let{statistics:r}=e.source,{systemMemoryBytes:n,gpuMemoryBytes:i}=e,s=gm(e.state,e.priorityTier);r[s*An+$i.numChunks]+=t,r[s*An+$i.systemMemoryBytes]+=t*n,r[s*An+$i.gpuMemoryBytes]+=t*i}var Ne=class extends Hi{constructor(t,r){let n=t.get(r.chunkManager);super(n),Ml(this,t,r)}};function bC(e){let t=e.downloadAbortController=new AbortController,r=Date.now();e.source.download(e,t.signal).then(()=>{if(e.downloadAbortController===t){e.downloadAbortController=void 0;let n=Date.now(),{statistics:i}=e.source;i[Tl(io.totalTime)]+=n-r,++i[Tl(io.totalChunks)],e.downloadSucceeded()}},n=>{e.downloadAbortController===t&&(e.downloadAbortController=void 0,e.downloadFailed(n),console.log(`Error retrieving chunk ${e}: ${n}`))})}function fg(e){let t=e.downloadAbortController;e.downloadAbortController=void 0,t.abort(new DOMException("chunk download cancelled","AbortError"))}var xo=class{constructor(t,r){this.heapOperations=t,this.linkedListOperations=r,r.initializeHead(this.recentHead)}heapRoots=[null,null];recentHead=new fe;add(t){let r=t.priorityTier;if(r===se.RECENT)this.linkedListOperations.insertAfter(this.recentHead,t);else{let{heapRoots:n}=this;n[r]=this.heapOperations.meld(n[r],t)}}*candidates(){if(this.heapOperations.compare===fe.priorityLess){let{linkedListOperations:t,recentHead:r}=this;for(;;){let i=t.back(r);if(i==null)break;yield i}let{heapRoots:n}=this;for(let i=se.LAST_ORDERED_TIER;i>=se.FIRST_ORDERED_TIER;--i)for(;;){let s=n[i];if(s==null)break;yield s}}else{let t=this.heapRoots;for(let i=se.FIRST_ORDERED_TIER;i<=se.LAST_ORDERED_TIER;++i)for(;;){let s=t[i];if(s==null)break;yield s}let{linkedListOperations:r,recentHead:n}=this;for(;;){let i=r.front(n);if(i==null)break;yield i}}}delete(t){let r=t.priorityTier;if(r===se.RECENT)this.linkedListOperations.pop(t);else{let n=this.heapRoots;n[r]=this.heapOperations.remove(n[r],t)}}},IC=so({next:"next0",prev:"prev0"}),_C=so({next:"next1",prev:"prev1"});function CC(e){return new xo(vo({compare:e,child:"child0",next:"next0",prev:"prev0"}),IC)}function Nr(e){return new xo(vo({compare:e,child:"child1",next:"next1",prev:"prev1"}),_C)}function Hl(e,t,r,n,i,s){for(;t.availableItems<1||t.availableSize<e;){let o=i.next().value;if(o===void 0)return!1;let a=o.priorityTier;if(a<r||a===r&&o.priority>=n)return!1;s(o)}return!0}var Wl=class extends be{constructor(t,r){super(),this.itemLimit=t,this.sizeLimit=r,this.registerDisposer(t.changed.add(this.capacityChanged.dispatch)),this.registerDisposer(r.changed.add(this.capacityChanged.dispatch))}currentSize=0;currentItems=0;capacityChanged=new $e;adjust(t,r){this.currentItems-=t,this.currentSize-=r}get availableSize(){return this.sizeLimit.value-this.currentSize}get availableItems(){return this.itemLimit.value-this.currentItems}toString(){return`bytes=${this.currentSize}/${this.sizeLimit.value},items=${this.currentItems}/${this.itemLimit.value}`}},hg=class extends me{gpuMemoryCapacity;systemMemoryCapacity;downloadCapacity;computeCapacity;enablePrefetch;sources=new Set;queuedDownloadPromotionQueue=[Nr(fe.priorityGreater),Nr(fe.priorityGreater)];queuedComputePromotionQueue=Nr(fe.priorityGreater);downloadEvictionQueue=[Nr(fe.priorityLess),Nr(fe.priorityLess)];computeEvictionQueue=Nr(fe.priorityLess);systemMemoryEvictionQueue=CC(fe.priorityLess);gpuMemoryPromotionQueue=Nr(fe.priorityGreater);gpuMemoryEvictionQueue=Nr(fe.priorityLess);updatePending=null;gpuMemoryChanged=new $e;numQueued=0;numFailed=0;gpuMemoryGeneration=0;constructor(e,t){super(e,t);let r=n=>{let i=this.registerDisposer(new Wl(e.get(n.itemLimit),e.get(n.sizeLimit)));return i.capacityChanged.add(()=>this.scheduleUpdate()),i};this.gpuMemoryCapacity=r(t.gpuMemoryCapacity),this.systemMemoryCapacity=r(t.systemMemoryCapacity),this.enablePrefetch=e.get(t.enablePrefetch),this.downloadCapacity=[r(t.downloadCapacity),r(t.downloadCapacity)],this.computeCapacity=r(t.computeCapacity)}scheduleUpdate(){this.updatePending===null&&(this.updatePending=setTimeout(this.process.bind(this),0))}*chunkQueuesForChunk(e){switch(e.state){case z.QUEUED:e.isComputational?yield this.queuedComputePromotionQueue:yield this.queuedDownloadPromotionQueue[e.source.sourceQueueLevel];break;case z.DOWNLOADING:e.isComputational?yield this.computeEvictionQueue:(yield this.downloadEvictionQueue[e.source.sourceQueueLevel],yield this.systemMemoryEvictionQueue);break;case z.SYSTEM_MEMORY_WORKER:case z.SYSTEM_MEMORY:yield this.systemMemoryEvictionQueue,e.requestedState===z.GPU_MEMORY&&(yield this.gpuMemoryPromotionQueue);break;case z.GPU_MEMORY:yield this.systemMemoryEvictionQueue,yield this.gpuMemoryEvictionQueue;break}}adjustCapacitiesForChunk(e,t){let r=t?-1:1;switch(e.state){case z.FAILED:this.numFailed-=r;break;case z.QUEUED:this.numQueued-=r;break;case z.DOWNLOADING:(e.isComputational?this.computeCapacity:this.downloadCapacity[e.source.sourceQueueLevel]).adjust(r*e.downloadSlots,r*e.systemMemoryBytes),this.systemMemoryCapacity.adjust(r,r*e.systemMemoryBytes);break;case z.SYSTEM_MEMORY:case z.SYSTEM_MEMORY_WORKER:this.systemMemoryCapacity.adjust(r,r*e.systemMemoryBytes);break;case z.GPU_MEMORY:this.systemMemoryCapacity.adjust(r,r*e.systemMemoryBytes),this.gpuMemoryCapacity.adjust(r,r*e.gpuMemoryBytes);break}}removeChunkFromQueues_(e){vr(e,-1);for(let t of this.chunkQueuesForChunk(e))t.delete(e)}addChunkToQueues_(e){if(e.state===z.QUEUED&&e.priorityTier===se.RECENT){let{source:t}=e;return t.removeChunk(e),this.adjustCapacitiesForChunk(e,!1),!1}vr(e,1);for(let t of this.chunkQueuesForChunk(e))t.add(e);return!0}performChunkPriorityUpdate(e){if(e.priorityTier===e.newPriorityTier&&e.priority===e.newPriority){e.newPriorityTier=se.RECENT,e.newPriority=Number.NEGATIVE_INFINITY;return}Ji&&console.log(`${e}: changed priority ${e.priorityTier}:${e.priority} -> ${e.newPriorityTier}:${e.newPriority}`),this.removeChunkFromQueues_(e),e.updatePriorityProperties(),e.state===z.NEW&&(e.state=z.QUEUED,this.adjustCapacitiesForChunk(e,!0)),this.addChunkToQueues_(e)}updateChunkState(e,t){t!==e.state&&(Ji&&console.log(`${e}: changed state ${z[e.state]} -> ${z[t]}`),this.adjustCapacitiesForChunk(e,!1),this.removeChunkFromQueues_(e),e.state=t,this.adjustCapacitiesForChunk(e,!0),this.addChunkToQueues_(e),this.scheduleUpdate())}markRecentlyUsed(e){this.removeChunkFromQueues_(e),this.addChunkToQueues_(e)}processGPUPromotions_(){let e=this;function t(s){e.freeChunkGPUMemory(s),s.source.chunkManager.queueManager.updateChunkState(s,z.SYSTEM_MEMORY)}let r=this.gpuMemoryPromotionQueue.candidates(),n=this.gpuMemoryEvictionQueue.candidates(),i=this.gpuMemoryCapacity;for(;;){let s=r.next().value;if(s===void 0)break;let o=s.priorityTier,a=s.priority;if(!Hl(s.gpuMemoryBytes,i,o,a,n,t))break;this.copyChunkToGPU(s),this.updateChunkState(s,z.GPU_MEMORY)}}freeChunkGPUMemory(e){++this.gpuMemoryGeneration,this.rpc.invoke("Chunk.update",{id:e.key,state:z.SYSTEM_MEMORY,source:e.source.rpcId})}freeChunkSystemMemory(e){e.state===z.SYSTEM_MEMORY_WORKER?e.freeSystemMemory():this.rpc.invoke("Chunk.update",{id:e.key,state:z.EXPIRED,source:e.source.rpcId})}retrieveChunkData(e){return this.rpc.promiseInvoke("Chunk.retrieve",{key:e.key,source:e.source.rpcId})}copyChunkToGPU(e){++this.gpuMemoryGeneration;let t=this.rpc;if(e.state===z.SYSTEM_MEMORY)t.invoke("Chunk.update",{id:e.key,source:e.source.rpcId,state:z.GPU_MEMORY});else{let r={},n=[];e.serialize(r,n),r.state=z.GPU_MEMORY,t.invoke("Chunk.update",r,n)}}moveChunkToFrontend(e){let t=this.rpc,r={},n=[];e.serialize(r,n),r.state=z.SYSTEM_MEMORY,t.invoke("Chunk.update",r,n)}processQueuePromotions_(){let e=r=>{switch(r.state){case z.DOWNLOADING:fg(r);break;case z.GPU_MEMORY:this.freeChunkGPUMemory(r);case z.SYSTEM_MEMORY_WORKER:case z.SYSTEM_MEMORY:this.freeChunkSystemMemory(r);break}this.updateChunkState(r,z.QUEUED)},t=(r,n,i)=>{let s=this.systemMemoryEvictionQueue.candidates(),o=this.systemMemoryCapacity;for(;;){let a=r.next();if(a.done)return;let c=a.value,l=0,u=c.priorityTier,f=c.priority;if(!Hl(l,i,u,f,n,e)||!Hl(l,o,u,f,s,e))return;this.updateChunkState(c,z.DOWNLOADING),bC(c)}};for(let r=0;r<EC;++r)t(this.queuedDownloadPromotionQueue[r].candidates(),this.downloadEvictionQueue[r].candidates(),this.downloadCapacity[r]);t(this.queuedComputePromotionQueue.candidates(),this.computeEvictionQueue.candidates(),this.computeCapacity)}process(){if(!this.updatePending)return;this.updatePending=null;let e=this.gpuMemoryGeneration;this.processGPUPromotions_(),this.processQueuePromotions_(),this.logStatistics(),this.gpuMemoryGeneration!==e&&this.gpuMemoryChanged.dispatch()}logStatistics(){Ji&&console.log(`[Chunk status] QUEUED: ${this.numQueued}, FAILED: ${this.numFailed}, DOWNLOAD: ${this.downloadCapacity}, MEM: ${this.systemMemoryCapacity}, GPU: ${this.gpuMemoryCapacity}`)}invalidateSourceCache(e){for(let t of e.chunks.values()){switch(t.state){case z.DOWNLOADING:fg(t);break;case z.SYSTEM_MEMORY_WORKER:t.freeSystemMemory();break}this.updateChunkState(t,z.QUEUED)}this.rpc.invoke("Chunk.update",{source:e.rpcId}),this.scheduleUpdate()}};hg=Xl([j(vm)],hg);var Dr=class extends me{chunkManagerGeneration=-1;numVisibleChunksNeeded=0;numVisibleChunksAvailable=0;numPrefetchChunksNeeded=0;numPrefetchChunksAvailable=0},pg=200,dg=class extends me{queueManager;existingTierChunks=[];newTierChunks=[];updatePending=null;recomputeChunkPriorities=new $e;recomputeChunkPrioritiesLate=new $e;memoize=new Tn;layers=[];sendLayerChunkStatistics=this.registerCancellable(Ws(()=>{this.rpc.invoke(Em,{id:this.rpcId,layers:this.layers.map(e=>({id:e.rpcId,numVisibleChunksAvailable:e.numVisibleChunksAvailable,numVisibleChunksNeeded:e.numVisibleChunksNeeded,numPrefetchChunksAvailable:e.numPrefetchChunksAvailable,numPrefetchChunksNeeded:e.numPrefetchChunksNeeded}))})},pg));constructor(e,t){super(e,t),this.queueManager=e.get(t.chunkQueueManager).addRef(),this.registerDisposer(this.queueManager.gpuMemoryChanged.add(this.registerCancellable(Ws(()=>this.scheduleUpdateChunkPriorities(),pg,{leading:!1,trailing:!0}))));for(let r=se.FIRST_TIER;r<=se.LAST_TIER;++r)r!==se.RECENT&&(this.existingTierChunks[r]=[])}scheduleUpdateChunkPriorities(){this.updatePending===null&&(this.updatePending=setTimeout(this.recomputeChunkPriorities_.bind(this),0))}registerLayer(e){let t=this.recomputeChunkPriorities.count;e.chunkManagerGeneration!==t&&(e.chunkManagerGeneration=t,this.layers.push(e),e.numVisibleChunksAvailable=0,e.numVisibleChunksNeeded=0,e.numPrefetchChunksAvailable=0,e.numPrefetchChunksNeeded=0)}recomputeChunkPriorities_(){this.updatePending=null,this.layers.length=0,this.recomputeChunkPriorities.dispatch(),this.recomputeChunkPrioritiesLate.dispatch(),this.updateQueueState([se.VISIBLE,se.PREFETCH]),this.sendLayerChunkStatistics()}requestChunk(e,t,r,n=z.GPU_MEMORY){if(Number.isNaN(r))return;if(t===se.RECENT)throw new Error("Not going to request a chunk with the RECENT tier");e.newRequestedState=Math.min(e.newRequestedState,n),e.newPriorityTier===se.RECENT&&this.newTierChunks.push(e);let i=e.newPriorityTier;(t<i||t===i&&r>e.newPriority)&&(e.newPriorityTier=t,e.newPriority=r)}updateQueueState(e){let t=this.existingTierChunks,r=this.queueManager;for(let i of e){let s=t[i];Ji&&console.log(`existingTierChunks[${se[i]}].length=${s.length}`);for(let o of s)o.newPriorityTier===se.RECENT&&r.performChunkPriorityUpdate(o);s.length=0}let n=this.newTierChunks;for(let i of n)r.performChunkPriorityUpdate(i),t[i.priorityTier].push(i);Ji&&console.log(`updateQueueState: newTierChunks.length = ${n.length}`),n.length=0,this.queueManager.scheduleUpdate()}};dg=Xl([j(xm)],dg);function ye(e,t){let r=class extends e{parameters;constructor(...n){super(...n);let i=n[1];this.parameters=i.parameters}};return r=Xl([hm(t.RPC_ID)],r),r}function Xe(e){return class extends e{chunkManager;constructor(...t){super(...t);let r=t[0],n=t[1];this.chunkManager=r.get(n.chunkManager)}}}X(Sm,function(e){let t=this.get(e.id);t.chunkManager.queueManager.invalidateSourceCache(t)});xt(wm,function(e){let t=this.get(e.queue),r=new Map;for(let n of t.sources)r.set(n.rpcId,n.statistics);return Promise.resolve({value:r})});var So=class extends be{};function Zl(e){let t,r;return async(n,i)=>((r===void 0||n!==void 0&&t?.generation===n.generation)&&(t=void 0,r=Yi(async s=>(t=await e(n,s),t))),r(i??{}))}var wo=class extends be{constructor(t){super(),this.base=t}memoize=new Tn;getCredentialsProvider(t,r){return this.memoize.get({key:t,parameters:r},()=>this.registerDisposer(this.base.getCredentialsProvider(t,r).addRef()))}};var gg="CredentialsProvider",yg="CredentialsProvider.get",vg="CredentialsManager",xg="CredentialsManager.get";var AC=Object.defineProperty,MC=Object.getOwnPropertyDescriptor,Eg=(e,t,r,n)=>{for(var i=n>1?void 0:n?MC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&AC(t,r,i),i};var Sg=class extends me{get=Zl((e,t)=>this.rpc.promiseInvoke(yg,{providerId:this.rpcId,invalidCredentials:e},{signal:t.signal,progressListener:t.progressListener}))};Sg=Eg([j(gg)],Sg);function Nn(){return e=>class extends e{credentialsProvider;constructor(...t){super(...t);let r=t[1];this.credentialsProvider=this.rpc.getOptionalRef(r.credentialsProvider)}}}var Ql=class extends So{constructor(t,r,n,i){super(),this.rpc=t,this.managerId=r,this.key=n,this.parameters=i}get=Zl((t,r)=>this.rpc.promiseInvoke(xg,{managerId:this.managerId,key:this.key,parameters:this.parameters,invalidCredentials:t},{signal:r.signal,progressListener:r.progressListener}))},wg=class extends me{impl=new wo(this.makeBaseCredentialsManager());makeBaseCredentialsManager(){return{getCredentialsProvider:(e,t)=>new Ql(this.rpc,this.rpcId,e,t)}}getCredentialsProvider(e,t){return this.impl.getCredentialsProvider(e,t)}};wg=Eg([j(vg)],wg);function St(e,t){return e<t?-1:e>t?1:0}var Wi=class extends Error{constructor(t,r){super(`${t.getUrl()} not found`,r)}};async function Jt(e,t,r={}){return Pr(new ke(e,t),r)}async function Pr(e,t={}){let r=await e.read(t);if(t?.throwIfMissing===!0&&r===void 0)throw new Wi(e);if(t?.strictByteRange===!0&&r!==void 0){let{byteRange:n}=t,{offset:i,length:s}=r;if(n!==void 0&&("suffixLength"in n?s!==n.suffixLength:i!==n.offset||s!==void 0&&s!==n.length))throw new Error(`Received truncated response for ${e.getUrl()}, expected ${JSON.stringify(n)} but received offset=${i}, length=${s}`)}return r}function kC(e,t,r,n){switch(n){case"suffix":{let i=t.length;return{directories:e.directories.map(s=>s.substring(i)),entries:e.entries.map(({key:s,...o})=>({...o,key:s.substring(i)}))}}case"url":return{directories:e.directories.map(i=>r.getUrl(i)),entries:e.entries.map(({key:i,...s})=>({...s,key:r.getUrl(i)}))};default:return e}}async function Xi(e,t,r={}){if(!e.list)throw new Error("Listing not supported");return kC(await e.list(t,r),t,e,r.responseKeys)}var ke=class{constructor(t,r){this.store=t,this.key=r}stat(t){return this.store.stat(this.key,t)}read(t){return this.store.read(this.key,t)}getUrl(){return this.store.getUrl(this.key)}};function Eo(e){return e.entries.sort(({key:t},{key:r})=>St(t,r)),e.directories.sort(St),e}function TC(e){let t=e.match(/^((?:.*?\|)?)([a-zA-Z][a-zA-Z0-9-+.]*)(?:(:[^?#|]*)((?:[?#][^|]*)?))?$/);if(t===null)throw new Error(`Invalid URL: ${e}`);let[,r,n,i,s]=t;return i===void 0?`${r}${n}:`:i===":"||i.endsWith("/")?e:`${r}${n}${i}/${s??""}`}function bg(e){return e.match(/.*?([^|]*)$/)[1]}var NC=/^(?:([a-zA-Z][a-zA-Z0-9-+.]*):)?(.*)$/;function bo(e){let t=e.match(NC),r=t[1],n=t[2];return r===void 0?{url:e,scheme:e,suffix:void 0}:{url:e,scheme:r,suffix:n}}function Ig(e){return e.split("|").map(bo)}function Ve(e,...t){let[,r,n]=e.match(/^(.*?[^|?#]*)([^|]*)$/);for(let i of t)i.startsWith("/")&&(i=i.substring(1)),i!==""&&(r=TC(r),r+=Ze(i));return r+n}function eu(e,...t){for(let r of t)r.startsWith("/")&&(r=r.substring(1)),r!==""&&(e=nn(e),e+=r);return e}function nn(e){return Ag(e)||(e+="/"),e}function Rr(e){let{suffix:t}=e;if(t!==void 0&&t.match(/[#?]/))throw new Error(`Invalid URL ${e.url}: query parameters and/or fragment not supported`)}function _g(e){if(e.suffix)throw new Error(`Invalid URL syntax ${JSON.stringify(e.url)}, expected "${e.scheme}:"`)}function DC(e){let[,t,r]=e.match(/^(.*?[^|?#]*)([^|]*)$/);return{base:t,queryAndFragment:r}}function Cg(e,t){let r=e;e.endsWith("/")&&(e=e.substring(0,e.length-1));for(let n of t.split("/"))if(!(n===""||n===".")){if(n===".."){let i=e.lastIndexOf("/");if(i<=0)throw new Error(`Invalid relative path ${JSON.stringify(t)} from base path ${JSON.stringify(r)}`);e=e.substring(0,i);continue}e!==""&&(e+="/"),e+=n}return t.endsWith("/")&&(e+="/"),e}function Ag(e){return e===""||e.endsWith("/")}function Ze(e){return encodeURI(e).replace(/[?#@]/g,t=>`%${t.charCodeAt(0).toString(16).toUpperCase()}`)}function Ht(e,t){let{base:r,queryAndFragment:n}=DC(e);return r+Ze(t)+n}function Dn(e){let t=new URL(e);if(t.hash)throw new Error("fragment not supported");if(t.username||t.password)throw new Error("basic auth credentials not supported");return{baseUrl:`${t.origin}/${t.search}`,path:decodeURIComponent(t.pathname.substring(1))}}function Mg(e){return async t=>{let r=[],n=await Promise.allSettled(e.map(i=>i.match(t)));for(let i of n)i.status==="fulfilled"&&r.push(...i.value);return r}}function PC(e){let t=new Set,r=new Set;for(let n of e){let{fileNames:i,subDirectories:s}=n;if(i!==void 0)for(let o of i)t.add(o);if(s!==void 0)for(let o of s)r.add(o)}return{fileNames:t,subDirectories:r,match:Mg(e)}}function RC(e){let t=0,r=0;for(let n of e)t=Math.max(t,n.prefixLength),r=Math.max(r,n.suffixLength);return{prefixLength:t,suffixLength:r,match:Mg(e)}}var Pn=class{directorySpecs=[];fileSpecs=[];_directorySpec;_fileSpec;registerDirectoryFormat(t){this.directorySpecs.push(t),this._directorySpec=void 0}registerFileFormat(t){this.fileSpecs.push(t),this._fileSpec=void 0}copyTo(t){t.directorySpecs.push(...this.directorySpecs),t.fileSpecs.push(...this.fileSpecs),t._fileSpec=void 0,t._directorySpec=void 0}get directorySpec(){return this._directorySpec??(this._directorySpec=this.getDirectorySpec())}getDirectorySpec(){return PC(this.directorySpecs)}get fileSpec(){return this._fileSpec??(this._fileSpec=this.getFileSpec())}getFileSpec(){let{fileSpecs:t}=this,r=[...t];return RC(r)}};var Io=class{baseKvStoreProviders=new Map;kvStoreAdapterProviders=new Map;autoDetectRegistry=new Pn;getKvStore(t){let r=Ig(t),n;{let i=r[0];n=this.getBaseKvStoreProvider(i).getKvStore(i)}for(let i=1;i<r.length;++i)n=this.applyKvStoreAdapterUrl(n,r[i]);return n}getFileHandle(t){let{store:r,path:n}=this.getKvStore(t);return new ke(r,n)}getBaseKvStoreProvider(t){let r=this.baseKvStoreProviders.get(t.scheme);if(r===void 0){let n=this.describeProtocolUsage(t.scheme),i=`Invalid base kvstore protocol "${t.scheme}:"`;throw n!==void 0&&(i+=`; ${n}`),new Error(i)}return r}getKvStoreAdapterProvider(t){let r=this.kvStoreAdapterProviders.get(t.scheme);if(r===void 0){let n=this.describeProtocolUsage(t.scheme),i=`Invalid kvstore adapter protocol "${t.scheme}:"`;throw n!==void 0&&(i+=`; ${n}`),i+=`; supported schemes: ${JSON.stringify(Array.from(this.kvStoreAdapterProviders.keys()))}`,new Error(i)}return r}applyKvStoreAdapterUrl(t,r){return this.getKvStoreAdapterProvider(r).getKvStore(r,t)}describeProtocolUsage(t){if(this.baseKvStoreProviders.has(t))return`"${t}:" may only be used as a base kvstore protocol`;if(this.kvStoreAdapterProviders.has(t))return`"${t}:" may only be used as a kvstore adapter protocol`}stat(t,r={}){let n=this.getKvStore(t);return n.store.stat(n.path,r)}read(t,r={}){let n=this.getKvStore(t);return Jt(n.store,n.path,r)}list(t,r={}){let n=this.getKvStore(t);return Xi(n.store,n.path,r)}resolveRelativePath(t,r){let n=this.getKvStore(t);return n.store.getUrl(Cg(n.path,r))}};var Zi=class{baseKvStoreProviders=[];kvStoreAdapterProviders=[];autoDetectRegistry=new Pn;registerBaseKvStoreProvider(t){this.baseKvStoreProviders.push(t)}registerKvStoreAdapterProvider(t){this.kvStoreAdapterProviders.push(t)}applyToContext(t){let{kvStoreContext:r}=t;for(let n of["baseKvStoreProviders","kvStoreAdapterProviders"]){let i=r[n];for(let s of this[n]){let o=s(t),{scheme:a}=o;if(i.has(a))throw new Error(`Duplicate kvstore scheme ${a}`);i.set(a,o)}}this.autoDetectRegistry.copyTo(t.kvStoreContext.autoDetectRegistry)}},Wt=new Zi;var kg="SharedKvStoreContext",Tg="SharedKvStoreContext.stat",Ng="SharedKvStoreContext.read",tu="SharedKvStoreContext.list",Dg="SharedKvStoreContext.completeUrl";var OC=Object.defineProperty,UC=Object.getOwnPropertyDescriptor,LC=(e,t,r,n)=>{for(var i=n>1?void 0:n?UC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&OC(t,r,i),i};var Pg=class extends me{kvStoreContext;chunkManager;credentialsManager;constructor(e,t){super(e,t),this.chunkManager=e.get(t.chunkManager),this.credentialsManager=e.get(t.credentialsManager),this.kvStoreContext=new Io,Wt.applyToContext(this),wt.applyToContext(this)}};Pg=LC([j(kg)],Pg);var wt=new Zi;function Te(e){return class extends e{sharedKvStoreContext;constructor(...t){super(...t);let r=t[1];this.sharedKvStoreContext=this.rpc.get(r.sharedKvStoreContext)}}}var Rg="rendered_view.addLayer",Og="rendered_view.removeLayer",Ug="SharedProjectionParameters",Lg="SharedProjectionParameters.changed";var FC=Object.defineProperty,BC=Object.getOwnPropertyDescriptor,zC=(e,t,r,n)=>{for(var i=n>1?void 0:n?BC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&FC(t,r,i),i};var ru=class extends be{constructor(t){super(),this.view=t}state=void 0},ir=class extends Dr{attachments=new Map;attach(t){}};X(Rg,function(e){let t=this.get(e.view),r=this.get(e.layer),n=new ru(t);r.attachments.set(t,n),r.attach(n)});X(Og,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);r.attachments.delete(t),n.dispose()});var Fg=class extends me{value;oldValue;changed=new jt;constructor(e,t){super(e,t),this.value=t.value,this.oldValue=Object.assign({},this.value)}};Fg=zC([j(Ug)],Fg);X(Lg,function(e){let t=this.get(e.id),{value:r,oldValue:n}=t;Object.assign(n,r),Object.assign(r,e.value),t.changed.dispatch(n,r)});function $C(e,t,r){for(let n=0;n<r;++n){let i=t*n;e.fill(0,i,i+r),e[i+n]=1}return e}function nu(e,t,r=t){return $C(new e(t*r),t,Math.min(t,r))}function Bg(e,t,r,n,i,s){for(let o=0;o<s;++o){let a=o*n,c=o*t;for(let l=0;l<i;++l)e[c+l]=r[a+l]}return e}var Tt;function VC(e,t,r){let n=1;(Tt===void 0||Tt.length<r)&&(Tt=new Uint32Array(r));for(let i=0;i<r;++i)Tt[i]=i;for(let i=0;i<r;++i){let s=t*i,o=i;{let l=Math.abs(e[s+i]);for(let u=i+1;u<r;++u){let f=Math.abs(e[s+u]);f>l&&(l=f,o=u)}}if(i!==o){n*=-1;for(let l=0;l<r;++l){let u=t*l,f=e[u+i];e[u+i]=e[u+o],e[u+o]=f}{let l=Tt[i];Tt[i]=Tt[o],Tt[o]=l}}let a=e[s+i],c=1/a;n*=a;for(let l=0;l<r;++l)e[t*l+i]*=c;e[s+i]=c;for(let l=0;l<r;++l){if(l===i)continue;let u=-e[t*i+l];for(let f=0;f<r;++f){let h=t*f;e[h+l]+=u*e[h+i]}e[t*i+l]=u*c}}for(let i=0;i<r;++i){let s=Tt[i];for(;s!==i;){let o=t*i,a=t*s;for(let l=0;l<r;++l){let u=o+l,f=a+l,h=e[u];e[u]=e[f],e[f]=h}let c=Tt[i]=Tt[s];Tt[s]=s,s=c}}return n}function iu(e,t,r,n,i){return Bg(e,t,r,n,i,i),VC(e,t,i)}var jC=[{prefix:"Y",exponent:24,longPrefix:"yotta"},{prefix:"Z",exponent:21,longPrefix:"zetta"},{prefix:"E",exponent:18,longPrefix:"exa"},{prefix:"P",exponent:15,longPrefix:"peta"},{prefix:"T",exponent:12,longPrefix:"tera"},{prefix:"G",exponent:9,longPrefix:"giga"},{prefix:"M",exponent:6,longPrefix:"mega"},{prefix:"k",exponent:3,longPrefix:"kilo"},{prefix:"",exponent:0,longPrefix:""},{prefix:"m",exponent:-3,longPrefix:"milli"},{prefix:"\xB5",exponent:-6,longPrefix:"micro"},{prefix:"n",exponent:-9,longPrefix:"nano"},{prefix:"p",exponent:-12,longPrefix:"pico"},{prefix:"f",exponent:-15,longPrefix:"femto"},{prefix:"a",exponent:-18,longPrefix:"atto"},{prefix:"z",exponent:-21,longPrefix:"zepto"},{prefix:"y",exponent:-24,longPrefix:"yocto"}],su=[...jC,{prefix:"h",exponent:2,longPrefix:"hecto"},{prefix:"da",exponent:1,longPrefix:"deca"},{prefix:"d",exponent:-1,longPrefix:"deci"},{prefix:"c",exponent:-2,longPrefix:"centi"}],GC=[{prefix:"u",exponent:-6},...su],ou=new Map;ou.set("",{unit:"",exponent:0});var qC=new Map;for(let{prefix:e,exponent:t}of GC){qC.set(t,e);for(let r of["m","s","Hz","rad/s"])ou.set(`${e}${r}`,{unit:r,exponent:t})}function au(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=t[i]+r[i];return e}function zg(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=t[i]*r[i];return e}function Qi(e){let t=1;for(let r=0,n=e.length;r<n;++r)t*=e[r];return t}function cu(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=Math.min(t[i],r[i]);return e}function $g(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=Math.max(t[i],r[i]);return e}var Vg=new Float32Array(0),lu=new Float64Array(0),JO=Float64Array.of(1,1,1);function Gg(e){let{names:t,units:r,scales:n}=e,{valid:i=!0,rank:s=t.length,timestamps:o=t.map(()=>Number.NEGATIVE_INFINITY),ids:a=t.map((f,h)=>-h),boundingBoxes:c=[]}=e,{coordinateArrays:l=new Array(s)}=e,{bounds:u=XC(c,s)}=e;return{valid:i,rank:s,names:t,timestamps:o,ids:a,units:r,scales:n,boundingBoxes:c,bounds:u,coordinateArrays:l}}var HC=Gg({valid:!1,names:[],units:[],scales:lu,boundingBoxes:[]}),qg=Gg({valid:!0,names:[],units:[],scales:lu,boundingBoxes:[]});function WC(e,t,r){let{box:{lowerBounds:n,upperBounds:i},transform:s}=e,o=n.length,a=r,c=s[a*o+t],l=c,u=c,f=!1;for(let h=0;h<o;++h){let p=s[a*h+t];if(p===0)continue;let d=p*n[h],m=p*i[h];l+=Math.min(d,m),u+=Math.max(d,m),f=!0}if(f)return{lower:l,upper:u}}var _o=.001;function XC(e,t){let r=new Float64Array(t),n=new Float64Array(t);r.fill(Number.NEGATIVE_INFINITY),n.fill(Number.POSITIVE_INFINITY);let i=new Array(t);i.fill(0);let s=new Array(t);s.fill(0);for(let a of e)for(let c=0;c<t;++c){let l=WC(a,c,t);if(l===void 0)continue;let{lower:u,upper:f}=l;if(Number.isFinite(u)&&Number.isFinite(f)){let h,p,d,m;Math.abs(u-(h=Math.round(u)))<_o&&Math.abs(f-(p=Math.round(f)))<_o?(++s[c],u=h,f=p):Math.abs(u-(d=Math.floor(u))-.5)<_o&&Math.abs(f-(m=Math.floor(f))-.5)<_o&&(++i[c],u=d+.5,f=m+.5)}r[c]=r[c]===Number.NEGATIVE_INFINITY?u:Math.min(r[c],u),n[c]=n[c]===Number.POSITIVE_INFINITY?f:Math.max(n[c],f)}let o=s.map((a,c)=>i[c]>0&&a===0);return{lowerBounds:r,upperBounds:n,voxelCenterAtIntegerCoordinates:o}}var DU=B.create(),PU=Qt.create();function QC(e,t){return qt(e.globalDimensionNames,t.globalDimensionNames)&&qt(e.displayDimensionIndices,t.displayDimensionIndices)&&qt(e.canonicalVoxelFactors,t.canonicalVoxelFactors)&&qt(e.voxelPhysicalScales,t.voxelPhysicalScales)&&e.canonicalVoxelPhysicalSize===t.canonicalVoxelPhysicalSize&&qt(e.displayDimensionUnits,t.displayDimensionUnits)&&qt(e.displayDimensionScales,t.displayDimensionScales)}function On(e,t){let r=e.displayDimensionRenderInfo;return r===t?!0:QC(r,t)?(e.displayDimensionRenderInfo=t,!0):!1}var S4={channelCoordinateSpace:qg,shape:new Uint32Array(0),numChannels:1,coordinates:new Uint32Array(0)};function Yg(e,t,r,n,i){let s=t.length,o=r.length,a=e.length,c=!0;for(let l=0;l<n;++l){let u=l,f=0;for(let h=0;h<s;++h)f+=i[u+h*n]*t[h];u+=s*n;for(let h=0;h<o;++h)f+=i[u+h*n]*r[h];f+=i[u+o*n],l<a?e[l]=f:(f<0||f>=1)&&(c=!1)}return c}function Jg(e,t,r){e.fill(0),e[15]=1;let n=!0,{displayDimensionIndices:i}=t,{globalToRenderLayerDimensions:s,modelToRenderLayerTransform:o}=r,a=r.rank;for(let c=0;c<3;++c){let l=i[c];if(l===-1){n=!1;continue}let u=s[l];if(u===-1){n=!1;continue}e[c+12]=o[u+a*(a+1)];for(let f=0;f<3;++f)e[c+4*f]=o[u+(a+1)*f]}if(!n){let{globalDimensionNames:c}=t,l=Array.from(i.filter(u=>u!==-1),u=>c[u]).join(",\xA0");throw new Error(`Transform from model dimensions (${r.modelDimensionNames.join(",\xA0")}) to display dimensions (${l}) does not have full rank`)}}var Un=class e{size;transform;invTransform;detTransform;finiteRank;constructor(t,r,n){this.size=B.clone(t),this.transform=ge.clone(r),this.finiteRank=n;let i=ge.create(),s=iu(i,4,r,4,4);if(s===0)throw new Error("Transform is singular");this.invTransform=i,this.detTransform=s}toObject(){return{size:this.size,transform:this.transform,finiteRank:this.finiteRank}}static fromObject(t){return new e(t.size,t.transform,t.finiteRank)}globalToLocalSpatial(t,r){return B.transformMat4(t,r,this.invTransform)}localSpatialVectorToGlobal(t,r){return rg(t,r,this.transform)}globalToLocalNormal(t,r){return ng(t,r,this.transform)}};var W=(e=>(e[e.UINT8=0]="UINT8",e[e.INT8=1]="INT8",e[e.UINT16=2]="UINT16",e[e.INT16=3]="INT16",e[e.UINT32=4]="UINT32",e[e.INT32=5]="INT32",e[e.UINT64=6]="UINT64",e[e.FLOAT32=7]="FLOAT32",e))(W||{});var lt={0:1,1:1,2:2,3:2,4:4,5:4,6:8,7:4},uu={0:Uint8Array,1:Int8Array,2:Uint16Array,3:Int16Array,4:Uint32Array,5:Int32Array,6:BigUint64Array,7:Float32Array};function Co(e,t,r=0,n=t.byteLength){let i=lt[e];return new uu[e](t,r,n/i)}var Ur=!1,eA=!1,tA=ge.create();function rA(e,t){let r=0,n=Math.abs(e.detTransform),{transform:i,size:s}=e;for(let o=0;o<3;++o){let a=0;for(let l=0;l<3;++l)a+=t[l*4+2]*i[4*o+l];let c=s[o];r+=Math.abs(a)*c,n*=c}return n/r}function Hg(e,t,r){let{curPositionInChunks:n,fixedPositionWithinChunk:i}=e,{nonDisplayLowerClipBound:s,nonDisplayUpperClipBound:o}=e,{rank:a,chunkDataSize:c,lowerChunkBound:l,upperChunkBound:u}=e.source.spec;if(!Yg(n,t,r,e.layerRank,e.fixedLayerToChunkTransform))return!1;let f=.001;for(let h=0;h<a;++h){let p=n[h];if(p<s[h]-f||p>o[h]+f)return Ur&&console.log("excluding source",e,`because of chunkDim=${h}, sum=${p}`,s,o,e.fixedLayerToChunkTransform),!1;let d=c[h],m=n[h]=Math.min(u[h]-1,Math.max(l[h],Math.floor(p/d)));i[h]=p-m*d}return!0}function nA(e,t){let r=t.length,n=0;if(Ur&&console.log(t),r>1){let i=0;for(let s=0;s<r;++s){let o=t[s],{chunkLayout:a}=o,c=rA(a,e);Ur&&console.log(`chunksize = ${a.size}, sliceArea = ${c}`),c>i&&(i=c,n=s)}}return n}var Ln=new Un(B.create(),ge.create(),0);function iA(e,t){if(e.displayDimensionRenderInfo!==t.displayDimensionRenderInfo||e.pixelSize!==t.pixelSize)return!0;let{viewMatrix:r}=e,{viewMatrix:n}=t;for(let i=0;i<12;++i)if(r[i]!==n[i])return!0;return!1}var Ao=class extends rn{constructor(t){super(),this.projectionParameters=t,this.registerDisposer(t.changed.add((r,n)=>{iA(r,n)&&this.invalidateVisibleSources(),this.invalidateVisibleChunks()}))}visibleLayers=new Map;visibleSourcesStale=!0;invalidateVisibleSources(){this.visibleSourcesStale=!0}invalidateVisibleChunks(){}updateVisibleSources(){if(!this.visibleSourcesStale)return;this.visibleSourcesStale=!1;let t=this.projectionParameters.value.displayDimensionRenderInfo,{visibleLayers:r}=this;for(let[n,i]of r){let{allSources:s,visibleSources:o}=i;if(o.length=0,s.length===0||!On(i,t))continue;let a=nA(this.projectionParameters.value.viewMatrix,s.map(l=>l[0])),c=s[a];for(let l of n.filterVisibleSources(this,c))o.push(l);o.reverse(),Ur&&console.log("visible sources chosen",o)}}};function*Wg(e,t,r){let n=e.projectionParameters.value.pixelSize*1.1,i=r[0].effectiveVoxelSize,s=t.renderScaleTarget.value,o=u=>{let f=n*s;for(let h=0;h<3;++h){let p=u[h];if(p>f&&p>1.01*i[h])return!0}return!1},a=(u,f)=>{let h=n*s;for(let p=0;p<3;++p){let d=u[p],m=f[p];if(Math.abs(h-d)<Math.abs(h-m)&&d<1.01*m)return!0}return!1},c=r.length-1,l;for(Ur&&console.log(`Filtering ${r.length} visible sources`);;){let u=r[c];if(l!==void 0&&!a(u.effectiveVoxelSize,l)){Ur&&console.log(`  Stopping at ${c} because can't improve on prev voxel size: effectiveVoxelSize=${u.effectiveVoxelSize} prevVoxelSize=${l}`);break}if(yield u,c===0){Ur&&console.log("  Stopping because scaleIndex=0");break}if(!o(u.effectiveVoxelSize)){Ur&&console.log(`Stopping at at ${c} because can't improve on voxel size ${u.effectiveVoxelSize}`);break}l=u.effectiveVoxelSize,--c}}var Xg="SliceView",Zg="sliceview/RenderLayer",Qg="SliceView.addVisibleLayer",ey="SliceView.removeVisibleLayer",ty="ChunkManager.requestChunk",fu=new Float32Array(3),hu=new Float32Array(3),ry=ge.create(),ny=new Float32Array(24);function iy(e,t,r,n){let i=fu,s=hu,{lowerChunkDisplayBound:o,upperChunkDisplayBound:a}=t;for(let f=0;f<3;++f)i[f]=Math.max(i[f],o[f]),s[f]=Math.min(s[f],a[f]);let{curPositionInChunks:c,chunkDisplayDimensionIndices:l}=t;function u(){if(!n(i[0],i[1],i[2],s[0],s[1],s[2],e))return;let f=0,h=Math.max(0,s[0]-i[0]),p=h;for(let y=1;y<3;++y){let I=Math.max(0,s[y]-i[y]);p*=I,I>h&&(h=I,f=y)}if(p===0)return;if(p===1){c[l[0]]=i[0],c[l[1]]=i[1],c[l[2]]=i[2],r(i,e);return}let d=i[f],m=s[f],g=Math.floor(.5*(d+m));s[f]=g,u(),s[f]=m,i[f]=g,u(),i[f]=d}u()}function Mo(e,t,r,n){if(!Hg(r,e.globalPosition,t))return;let{size:i}=r.chunkLayout,s=ge.multiply(ry,e.viewProjectionMat,r.chunkLayout.transform);for(let l=0;l<3;++l){let u=i[l];for(let f=0;f<4;++f)s[4*l+f]*=u}let o=ny;ho(o,s);let a=fu,c=hu;a.fill(Number.NEGATIVE_INFINITY),c.fill(Number.POSITIVE_INFINITY),iy(o,r,n,po)}function ko(e,t,r,n,i){if(!Hg(r,e.globalPosition,t))return;let{size:s}=n,o=ge.multiply(ry,e.viewProjectionMat,n.transform);for(let d=0;d<3;++d){let m=s[d];for(let g=0;g<4;++g)o[4*d+g]*=m}let{upperChunkDisplayBound:a}=r,c=tA;ge.invert(c,o);let l=fu,u=hu,f=1e-4,h=.001;for(let d=0;d<3;++d){let m=c[12+d]+f/s[d],g=Math.abs(c[d]),y=Math.abs(c[4+d]),I=a[d],_=m-g-y;_>=I&&_<I+h?_=I-1:_=Math.floor(_),l[d]=_,u[d]=Math.floor(m+g+y+1)}let p=ny;for(let d=0;d<3;++d){let m=o[4*d],g=o[4*d+1],y=o[4*d+2];p[d]=m,p[4+d]=-m,p[8+d]=+g,p[12+d]=-g,p[16+d]=+y,p[20+d]=-y}{let m=o[12],g=o[4*3+1],y=o[4*3+2];p[3]=1+m,p[7]=1-m,p[11]=1+g,p[15]=1-g,p[19]=y,p[23]=-y}eA&&(console.log("clippingPlanes",p),console.log("modelViewProjection",o.join(",")),console.log(`lower=${l.join(",")}, upper=${u.join(",")}`)),iy(p,r,i,sg)}function To(e,t){let{finiteRank:r}=t;if(r===3)return t;Ln.finiteRank=r,B.copy(Ln.size,t.size);let n=ge.copy(Ln.transform,t.transform),i=ge.copy(Ln.invTransform,t.invTransform);Ln.detTransform=t.detTransform;let{invViewMatrix:s,width:o,height:a}=e,c=mo(e.projectionMat);for(let l=r;l<3;++l){let u=s[12+l],f=u,h=u,p=Math.abs(s[l]*o);f-=p,h+=p;let d=Math.abs(s[l+4]*a);f-=d,h+=d;let m=Math.abs(s[l+8]*c);f-=m,h+=m;let g=Math.max(1,h-f);n[12+l]=f,n[5*l]=g}return ge.invert(i,n),Ln}function sy(e){let t=.254829592,r=-.284496736,n=1.421413741,i=-1.453152027,s=1.061405429,a=1/(1+.3275911*Math.abs(e)),c=1-((((s*a+i)*a+n)*a+r)*a+t)*a*Math.exp(-e*e);return Math.sign(e)*c}var No=class{constructor(t=50,r=1e3){this.velocityHalfLifeMilliseconds=t,this.modelHalfLifeMilliseconds=r}lastTime=Number.NEGATIVE_INFINITY;rank=0;numSamples=0;prevPosition=new Float32Array;velocity=new Float32Array;mean=new Float32Array;variance=new Float32Array;reset(t){this.lastTime=Number.NEGATIVE_INFINITY,this.rank=t,this.numSamples=0,this.velocity=new Float32Array(t),this.prevPosition=new Float32Array(t),this.mean=new Float32Array(t),this.variance=new Float32Array(t)}addSample(t,r=Date.now()){let n=t.length;n!==this.rank&&this.reset(n);let i=this.numSamples;if(++this.numSamples,this.numSamples===0){this.prevPosition.set(t),this.lastTime=r;return}let s=r-this.lastTime;this.lastTime=r;let o=1-2**-(s/this.velocityHalfLifeMilliseconds),a=1-2**-(s/this.modelHalfLifeMilliseconds),{velocity:c,prevPosition:l,mean:u,variance:f}=this;for(let h=0;h<n;++h){let p=(t[h]-l[h])/Math.max(s,1);l[h]=t[h];let d=c[h],m=c[h]=d+o*(p-d);if(i===1)u[h]=m;else{let g=u[h],y=f[h],I=m-g;u[h]=g+a*I,f[h]=(1-a)*(y+a*I*I)}}}};function Et(e){return class extends e{visibility;constructor(...t){super(...t);let r=t[0],n=t[1];this.visibility=r.get(n.visibility),this.registerDisposer(this.visibility.changed.add(()=>this.chunkManager.scheduleUpdateChunkPriorities()))}}}function je(e){return e===Number.POSITIVE_INFINITY?se.VISIBLE:se.PREFETCH}function Ge(e){return e===Number.POSITIVE_INFINITY?0:e*ym}var sA=Object.defineProperty,oA=Object.getOwnPropertyDescriptor,py=(e,t,r,n)=>{for(var i=n>1?void 0:n?oA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&sA(t,r,i),i};var du=-1e12,ts=1e9,oy=B.create(),aA=B.create(),cA=B.create(),pu=class extends Ao{constructor(t,r){super(t.get(r.projectionParameters)),this.initializeSharedObject(t,r.id)}};function ay(e){for(let t of e)for(let r of t)r.source.dispose()}var lA=Et(Xe(pu)),cy=class extends lA{velocityEstimator=new No;constructor(e,t){super(e,t),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateVisibleChunks()})),this.registerDisposer(this.projectionParameters.changed.add(()=>{this.velocityEstimator.addSample(this.projectionParameters.value.globalPosition)}))}invalidateVisibleChunks(){super.invalidateVisibleChunks(),this.chunkManager.scheduleUpdateChunkPriorities()}handleLayerChanged=()=>{this.chunkManager.scheduleUpdateChunkPriorities()};updateVisibleChunks(){let e=this.projectionParameters.value,t=this.chunkManager,r=this.visibility.value;if(r===Number.NEGATIVE_INFINITY)return;this.updateVisibleSources();let{centerDataPosition:n}=e,i=je(r),s=Ge(r);s+=du;let o=aA,a=cA,c=[];this.velocityEstimator.addSample(this.projectionParameters.value.globalPosition);for(let[l,u]of this.visibleLayers){t.registerLayer(l);let{visibleSources:f}=u;for(let h=0,p=f.length;h<p;++h){let d=f[h],m=t.queueManager.enablePrefetch.value?fA(this.velocityEstimator,d):[],{chunkLayout:g}=d;g.globalToLocalSpatial(o,n);let{size:y,finiteRank:I}=g;B.copy(a,y);for(let C=I;C<3;++C)a[C]=0,o[C]=0;let E=s+ts*h;c.length=0;let b=mg();if(ko(e,d.renderLayer.localPosition.value,d,To(e,d.chunkLayout),C=>{B.multiply(oy,C,a);let v=-B.distance(o,oy),{curPositionInChunks:w}=d,x=d.source.getChunk(w);t.requestChunk(x,i,E+v),++l.numVisibleChunksNeeded,x.state===z.GPU_MEMORY&&++l.numVisibleChunksAvailable,c.push(x),x.markGeneration=b}),m.length!==0){let{curPositionInChunks:C}=d;for(let v of c){C.set(v.chunkGridPosition);for(let w=0,x=m.length;w<x;){let k=m[w],M=m[w+2],P=m[w+3],F=m[w+4],S=m[w+5],O=C[k],R=O+m[w+1];if(R<M||R>P){w=S;continue}C[k]=R;let N=d.source.getChunk(C);if(C[k]=O,N.markGeneration===b){w=S;continue}t.requestChunk(N,se.PREFETCH,E+F),++l.numPrefetchChunksNeeded,N.state===z.GPU_MEMORY&&++l.numPrefetchChunksAvailable,w+=es}}}}}}removeVisibleLayer(e){let{visibleLayers:t}=this,r=t.get(e);t.delete(e),ay(r.allSources),e.renderScaleTarget.changed.remove(this.invalidateVisibleSources),e.localPosition.changed.remove(this.handleLayerChanged),this.invalidateVisibleSources()}addVisibleLayer(e,t,r){let n=this.visibleLayers.get(e);n===void 0?(n={allSources:t,visibleSources:[],displayDimensionRenderInfo:r},this.visibleLayers.set(e,n),e.renderScaleTarget.changed.add(()=>this.invalidateVisibleSources()),e.localPosition.changed.add(this.handleLayerChanged)):(ay(n.allSources),n.allSources=t,n.visibleSources.length=0,n.displayDimensionRenderInfo=r),this.invalidateVisibleSources()}disposed(){for(let e of this.visibleLayers.keys())this.removeVisibleLayer(e);super.disposed()}invalidateVisibleSources(){super.invalidateVisibleSources(),this.chunkManager.scheduleUpdateChunkPriorities()}};cy=py([j(Xg)],cy);function sn(e,t,r){return t.map(i=>i.map(s=>{let o=e.getRef(s.source),a=s.chunkLayout,{rank:c}=o.spec;return{renderLayer:r,source:o,chunkLayout:Un.fromObject(a),layerRank:s.layerRank,nonDisplayLowerClipBound:s.nonDisplayLowerClipBound,nonDisplayUpperClipBound:s.nonDisplayUpperClipBound,lowerClipBound:s.lowerClipBound,upperClipBound:s.upperClipBound,lowerClipDisplayBound:s.lowerClipDisplayBound,upperClipDisplayBound:s.upperClipDisplayBound,lowerChunkDisplayBound:s.lowerChunkDisplayBound,upperChunkDisplayBound:s.upperChunkDisplayBound,effectiveVoxelSize:s.effectiveVoxelSize,chunkDisplayDimensionIndices:s.chunkDisplayDimensionIndices,fixedLayerToChunkTransform:s.fixedLayerToChunkTransform,combinedGlobalLocalToChunkTransform:s.combinedGlobalLocalToChunkTransform,curPositionInChunks:new Float32Array(c),fixedPositionWithinChunk:new Uint32Array(c)}}))}X(Qg,function(e){let t=this.get(e.id),r=this.get(e.layerId),n=sn(this,e.sources,r);t.addVisibleLayer(r,n,e.displayDimensionRenderInfo)});X(ey,function(e){let t=this.get(e.id),r=this.get(e.layerId);t.removeVisibleLayer(r)});var Fn=class extends fe{chunkGridPosition;source=null;initializeVolumeChunk(t,r){super.initialize(t),this.chunkGridPosition=Float32Array.from(r)}serialize(t,r){super.serialize(t,r),t.chunkGridPosition=this.chunkGridPosition}downloadSucceeded(){super.downloadSucceeded()}freeSystemMemory(){}toString(){return this.source.toString()+":"+tr(this.chunkGridPosition)}},Bn=class extends Ne{spec;constructor(t,r){super(t,r),this.spec=r.spec}getChunk(t){let r=t.join(),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(this.chunkConstructor),n.initializeVolumeChunk(r,t),this.addChunk(n)),n}},ly=class extends me{renderScaleTarget;localPosition;numVisibleChunksNeeded;numVisibleChunksAvailable;numPrefetchChunksNeeded;numPrefetchChunksAvailable;chunkManagerGeneration;constructor(e,t){super(e,t),this.renderScaleTarget=e.get(t.renderScaleTarget),this.localPosition=e.get(t.localPosition),this.numVisibleChunksNeeded=0,this.numVisibleChunksAvailable=0,this.numPrefetchChunksAvailable=0,this.numPrefetchChunksNeeded=0,this.chunkManagerGeneration=-1}filterVisibleSources(e,t){return Wg(e,this,t)}};ly=py([j(Zg)],ly);var uy=2e3,uA=.1,fy=32,hy=.05,es=6;function fA(e,t){let r=[],n=e.rank,{combinedGlobalLocalToChunkTransform:i,layerRank:s}=t,{rank:o,chunkDataSize:a}=t.source.spec,{mean:c,variance:l}=e;for(let u=0;u<o;++u){let f=t.chunkDisplayDimensionIndices.includes(u),h=0,p=0;for(let w=0;w<n;++w){let x=c[w],k=l[w],M=i[w*s+u];h+=M*x,p+=M*M*k}if(h>uA)continue;let d=a[u],m=f?0:t.fixedPositionWithinChunk[u]/d,g=h/d*uy,y=Math.sqrt(2*p)/d*uy;if(Math.abs(g)<.001&&y<.001)continue;y=Math.max(1e-6,y);let I=w=>.5*(1+sy((w-g)/y)),_=t.curPositionInChunks[u],E=Math.floor(t.lowerClipBound[u]/d),b=Math.ceil(t.upperClipBound[u]/d)-1,C=r.length;for(let w=1;w<=fy&&!(!f&&_+w>b);++w){let x=1-I(w-m);if(x<hy)break;r.push(u,w,E,b,x,0)}let v=r.length;for(let w=C,x=r.length;w<x;w+=es)r[w+es-1]=v;C=v;for(let w=1;w<=fy&&!(!f&&_-w<E);++w){let x=I(-w+1-m);if(x<hy)break;r.push(u,-w,E,b,x,0)}v=r.length;for(let w=C,x=r.length;w<x;w+=es)r[w+es-1]=v}return r}xt(ty,async function(e,t){let r=this.get(e.source),{chunkManager:n}=r,i=r.getChunk(e.chunkGridPosition),s=i.key;if(i.state<=z.SYSTEM_MEMORY)return{value:void 0};if(i.state===z.FAILED)throw i.error;let o=n.recomputeChunkPriorities.add(()=>{n.requestChunk(i,se.VISIBLE,Number.POSITIVE_INFINITY,z.SYSTEM_MEMORY)});n.scheduleUpdateChunkPriorities();let a,c=new Promise((l,u)=>{a=f=>{if(f.state===z.FAILED){u(f.error);return}f.state<=z.SYSTEM_MEMORY&&l()}});r.registerChunkListener(s,a);try{return await Qs(c,t.signal),{value:void 0}}finally{r.unregisterChunkListener(s,a),o(),n.scheduleUpdateChunkPriorities()}});var dy="perspective_view/PerspectiveView";var hA=Object.defineProperty,pA=Object.getOwnPropertyDescriptor,dA=(e,t,r,n)=>{for(var i=n>1?void 0:n?pA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&hA(t,r,i),i};var my=class extends me{visibility;projectionParameters;constructor(...e){super(...e);let t=e[0],r=e[1];this.visibility=t.get(r.visibility),this.projectionParameters=t.get(r.projectionParameters)}};my=dA([j(dy)],my);var rs=class extends ir{};var gy="volume_rendering/VolumeRenderingRenderLayer",yy="volume_rendering/VolumeRenderingRenderLayer/update",mA=!1,gA=kt.create();function vy(e,t,r,n,i,s){if(n.length===0)return;let{viewMatrix:o,projectionMat:a,displayDimensionRenderInfo:c}=e,{voxelPhysicalScales:l}=c,u=Ki(l),f=mo(a),p=(f/r)**3,d=kt.determinant(fo(gA,o)),m={spatialScales:new Map,activeIndex:-1},g=v=>{let w=n[v];return Math.abs(w.chunkLayout.detTransform*d)},y=n.length-1,I=g(y);for(let v=y;v>=0;--v){let w=g(v),x=Math.cbrt(w*u/d),k=f/Math.cbrt(w);m.spatialScales.set(x,k),w-p>=0&&(I=w,y=v),m.activeIndex=y}if(mA){console.log(n);for(let v=0;v<n.length;++v){let w=g(v),x=f/Math.cbrt(w);console.log(`scaleIndex=${v} viewVolume=${w} bestScaleIndex=${y} actualViewVolume=${p}, desiredSamples=${x}, difference=${w-p}`)}}let _=Math.cbrt(I*u/d),E=f/Math.cbrt(I),b=!0,C=n[y];Mo(e,t,C,(v,w)=>{b&&(i(C,y,_,E,w,m),b=!1),s(C,y,v)})}var yA=Object.defineProperty,vA=Object.getOwnPropertyDescriptor,xA=(e,t,r,n)=>{for(var i=n>1?void 0:n?vA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&yA(t,r,i),i};var xy=B.create(),SA=B.create(),wA=B.create(),EA=B.create(),Sy=class extends Xe(ir){localPosition;renderScaleTarget;constructor(e,t){super(e,t),this.renderScaleTarget=e.get(t.renderScaleTarget),this.localPosition=e.get(t.localPosition);let r=()=>this.chunkManager.scheduleUpdateChunkPriorities();this.registerDisposer(this.localPosition.changed.add(r)),this.registerDisposer(this.renderScaleTarget.changed.add(r)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(t),e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.state={displayDimensionRenderInfo:r.projectionParameters.value.displayDimensionRenderInfo,transformedSources:[]}}recomputeChunkPriorities(){for(let e of this.attachments.values()){let{view:t}=e,r=t.visibility.value;if(r===Number.NEGATIVE_INFINITY)continue;let n=e.state,{transformedSources:i}=n;if(i.length===0||!On(n,t.projectionParameters.value.displayDimensionRenderInfo))continue;let s=t.projectionParameters.value,o=je(r),a=Ge(r);a+=du;let c=SA,l=wA,u=EA,{globalPosition:f,displayDimensionRenderInfo:{displayDimensionIndices:h}}=s;for(let m=0;m<3;++m){let g=h[m];u[m]=g===-1?0:f[g]}let p,{chunkManager:d}=this;d.registerLayer(this),vy(s,this.localPosition.value,this.renderScaleTarget.value,i[0],(m,g)=>{let{chunkLayout:y}=m;y.globalToLocalSpatial(c,u);let{size:I,finiteRank:_}=y;B.copy(l,I);for(let b=_;b<3;++b)l[b]=0,c[b]=0;let E=i[0].length-1-g;p=a+ts*E},(m,g,y)=>{B.multiply(xy,y,l);let I=-B.distance(c,xy),_=m.source.getChunk(m.curPositionInChunks);++this.numVisibleChunksNeeded,d.requestChunk(_,o,p+I),_.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable})}}};Sy=xA([j(gy)],Sy);X(yy,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);n.state.transformedSources=sn(this,e.sources,r),n.state.displayDimensionRenderInfo=e.displayDimensionRenderInfo,r.chunkManager.scheduleUpdateChunkPriorities()});var wy="annotation.MetadataChunkSource";var Ey="annotation.SubsetGeometryChunkSource",by="annotation.reference.add",Iy="annotation.reference.delete",_y="annotation.commit",mu="annotation.commit",Cy="annotation/SpatiallyIndexedRenderLayer",Ay="annotation/PerspectiveRenderLayer:updateSources",My="annotation/RenderLayer",ky="annotation/RenderLayer.updateSegmentation",bA=kt.create();function Ty(e,t,r,n,i,s){let{displayDimensionRenderInfo:o,viewMatrix:a,projectionMat:c,width:l,height:u}=e,{voxelPhysicalScales:f}=o,h=Math.abs(kt.determinant(fo(bA,a))),p=Ki(f),d=og(c)/h*p;if(n.length===0)return;let m=n[0],g=Math.abs(m.chunkLayout.detTransform)*p,{lowerClipDisplayBound:y,upperClipDisplayBound:I}=m;for(let w=0;w<3;++w)g*=I[w]-y[w];let _=Math.min(g,d),E=l*u,C=E/r**2/_,v=0;for(let w=n.length-1;w>=0&&v<C;--w){let x=n[w],k=x.source.spec,{chunkLayout:M}=x,P=Ki(M.size)*Math.abs(M.detTransform)*p,{limit:F,rank:S}=k,{nonDisplayLowerClipBound:O,nonDisplayUpperClipBound:R}=x,N=1;for(let ie=0;ie<S;++ie){let Ce=R[ie]-O[ie];Number.isFinite(Ce)&&(N/=Ce)}let U=F*N/P,L=!0,G=v+U,V=(1/G)**(1/3),Y=Math.sqrt(E/(G*_)),J=(C-v)*P/N,de=Math.min(1,J/k.limit);Mo(e,t,x,()=>{L&&(i(x,w),L=!1),s(x,w,de,V,Y)}),v=G}}var Lr=(e=>(e[e.MIN_REPRESENTATIVE=0]="MIN_REPRESENTATIVE",e[e.MAX_REPRESENTATIVE=1]="MAX_REPRESENTATIVE",e[e.REPRESENTATIVE_EXCLUDED=2]="REPRESENTATIVE_EXCLUDED",e[e.NONREPRESENTATIVE_EXCLUDED=4]="NONREPRESENTATIVE_EXCLUDED",e))(Lr||{});var yu=class{constructor(t){this.value=t,this.min=t}rank=0;parent=this;next=this;prev=this;min};function gu(e){let t=e,r=e.parent;for(;r!==e;)e=r,r=e.parent;for(e=t.parent;r!==e;)t.parent=r,t=e,e=t.parent;return r}function IA(e,t){let r=e.rank,n=t.rank;return r>n?(t.parent=e,e):(e.parent=t,r===n&&(t.rank=n+1),t)}function _A(e,t){let r=e.prev,n=t.prev;t.prev=r,r.next=t,e.prev=n,n.next=e}function*Ny(e){let t=e;do yield t.value,t=t.next;while(t!==e)}function Dy(e){return e.parent===e}var Do=class{map=new Map;visibleSegmentEquivalencePolicy=new Gt(Lr.MIN_REPRESENTATIVE);generation=0;has(t){return this.map.has(t)}get(t){let r=this.map.get(t);return r===void 0?t:gu(r).min}isMinElement(t){return t===this.get(t)}makeSet(t){let{map:r}=this,n=r.get(t);return n===void 0?(n=new yu(t),r.set(t,n),n):gu(n)}link(t,r){let n=this.makeSet(t),i=this.makeSet(r);if(n===i)return!1;this.generation++;let s=IA(n,i);_A(n,i);let o=n.min,a=i.min,c=(this.visibleSegmentEquivalencePolicy.value&Lr.MAX_REPRESENTATIVE)!==0;return s.min=o<a===c?a:o,!0}linkAll(t){for(let r=1,n=t.length;r<n;++r)this.link(t[0],t[r])}deleteSet(t){let{map:r}=this,n=!1;for(let i of this.setElements(t))r.delete(i),n=!0;return n&&++this.generation,n}*setElements(t){let r=this.map.get(t);r===void 0?yield t:yield*Ny(r)}clear(){let{map:t}=this;return t.size===0?!1:(++this.generation,t.clear(),!0)}get size(){return this.map.size}*mappings(){for(let t of this.map.values())yield[t.value,gu(t).min]}*roots(){for(let t of this.map.values())Dy(t)&&(yield t.value)}[Symbol.iterator](){return this.mappings()}toJSON(){let t=new Array;for(let r of this.map.values())if(Dy(r)){let n=new Array;for(let i of Ny(r))n.push(i);n.sort(mr),t.push(n)}return t.sort((r,n)=>mr(r[0],n[0])),t.map(r=>r.map(n=>n.toString()))}};var CA=Object.defineProperty,AA=Object.getOwnPropertyDescriptor,MA=(e,t,r,n)=>{for(var i=n>1?void 0:n?AA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&CA(t,r,i),i};var kA="DisjointUint64Sets",Ry="DisjointUint64Sets.add",Oy="DisjointUint64Sets.clear",Uy="DisjointUint64Sets.highBitRepresentativeChanged",Ly="DisjointUint64Sets.deleteSet",Po=class extends me{disjointSets=new Do;changed=new $e;get value(){return this}static makeWithCounterpart(e,t){let r=new Po;return r.disjointSets.visibleSegmentEquivalencePolicy=t,r.registerDisposer(t.changed.add(()=>{Py(r)})),r.initializeCounterpart(e),t.value&&Py(r),r}link(e,t){if(this.disjointSets.link(e,t)){let{rpc:r}=this;return r&&r.invoke(Ry,{id:this.rpcId,a:e,b:t}),this.changed.dispatch(),!0}return!1}linkAll(e){for(let t=1,r=e.length;t<r;++t)this.link(e[0],e[t])}has(e){return this.disjointSets.has(e)}get(e){return this.disjointSets.get(e)}clear(){if(this.disjointSets.clear()){let{rpc:e}=this;e&&e.invoke(Oy,{id:this.rpcId}),this.changed.dispatch()}}setElements(e){return this.disjointSets.setElements(e)}deleteSet(e){if(this.disjointSets.deleteSet(e)){let{rpc:t}=this;t&&t.invoke(Ly,{id:this.rpcId,x:e}),this.changed.dispatch()}}get size(){return this.disjointSets.size}toJSON(){return this.disjointSets.toJSON()}restoreState(e){e!==void 0&&ct(e,t=>{let r;ct(t,n=>{let i=yr(n);r!==void 0&&this.link(r,i),r=i})})}assignFrom(e){this.clear(),e instanceof Po&&(e=e.disjointSets);for(let[t,r]of e)this.link(t,r)}};Po=MA([j(kA)],Po);X(Ry,function(e){let t=this.get(e.id);t.disjointSets.link(e.a,e.b)&&t.changed.dispatch()});X(Oy,function(e){let t=this.get(e.id);t.disjointSets.clear()&&t.changed.dispatch()});function Py(e){e.rpc.invoke(Uy,{id:e.rpcId,value:e.disjointSets.visibleSegmentEquivalencePolicy.value})}X(Uy,function(e){let t=this.get(e.id);t.disjointSets.visibleSegmentEquivalencePolicy.value=e.value});X(Ly,function(e){let t=this.get(e.id);t.disjointSets.deleteSet(e.x)&&t.changed.dispatch()});function zn(e,t){return t>>>=0,e>>>=0,t=Math.imul(t,3432918353)>>>0,t=(t<<15|t>>>17)>>>0,t=Math.imul(t,461845907)>>>0,e=(e^t)>>>0,e=(e<<13|e>>>19)>>>0,e=e*5+3864292196>>>0,e}function vu(e=128){let t=Math.ceil(e/32),r=new Uint32Array(t);crypto.getRandomValues(r);let n="";for(let i=0;i<t;++i)n+=("00000000"+r[i].toString(16)).slice(-8);return n}function Fy(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength),r=65536;for(let n=0,i=t.length;n<i;n+=r)crypto.getRandomValues(t.subarray(n,Math.min(i,n+r)));return e}var By=3,TA=.8,Fr=!1,sr=0n,zy=0n,Ro=class e{constructor(t=e.generateHashSeeds(By)){this.hashSeeds=t;let r=8;for(;r<2*t.length;)r*=2;this.allocate(r)}loadFactor=TA;size=0;table;tableSize;empty=0xffffffffffffffffn;maxRehashAttempts=5;maxAttempts=5;capacity;generation=0;mungedEmptyKey;updateHashFunctions(t){this.hashSeeds=e.generateHashSeeds(t),this.mungedEmptyKey=void 0}tableWithMungedEmptyKey(t){let r=this.hashSeeds.length,n=new Array(r);for(let a=0;a<r;++a)n[a]=this.getHash(a,this.empty);let{mungedEmptyKey:i}=this;if(i===void 0)e:for(;;){i=Nl();for(let a=0;a<r;++a){let c=this.getHash(a,i);for(let l=0;l<r;++l)if(n[l]===c)continue e}this.mungedEmptyKey=i;break}let{table:s,empty:o}=this;for(let a=0;a<r;++a){let c=n[a];s[c]===o&&(s[c]=i)}try{t(s)}finally{for(let a=0;a<r;++a){let c=n[a];s[c]===i&&(s[c]=o)}}}static generateHashSeeds(t=By){return Fy(new Uint32Array(t))}getHash(t,r){let n=this.hashSeeds[t];return n=zn(n,Number(r&0xffffffffn)),n=zn(n,Number(r>>32n)),this.entryStride*(n&this.tableSize-1)}*keys(){let{empty:t,entryStride:r}=this,{table:n}=this;for(let i=0,s=n.length;i<s;i+=r){let o=n[i];o!==t&&(yield o)}}indexOf(t){let{table:r,empty:n}=this;if(t===n)return-1;for(let i=0,s=this.hashSeeds.length;i<s;++i){let o=this.getHash(i,t);if(r[o]===t)return o}return-1}chooseAnotherEmptyKey(){let{empty:t,table:r,entryStride:n}=this,i;for(;i=Nl(),!(i!==t&&!this.has(i)););this.empty=i;for(let s=0,o=r.length;s<o;s+=n)r[s]===t&&(r[s]=i)}has(t){return this.indexOf(t)!==-1}delete(t){let r=this.indexOf(t);if(r!==-1){let{table:n}=this;return n[r]=this.empty,++this.generation,this.size--,!0}return!1}clearTable(){let{table:t,empty:r}=this;t.fill(r)}clear(){return this.size===0?!1:(this.size=0,++this.generation,this.clearTable(),!0)}reserve(t){return t>this.capacity?(this.backupPending(),this.grow(t),this.restorePending(),!0):!1}swapPending(t,r){let n=sr;this.storePending(t,r),t[r]=n}storePending(t,r){sr=t[r]}backupPending(){zy=sr}restorePending(){sr=zy}tryToInsert(){Fr&&console.log(`tryToInsert: ${sr}`);let t=0,{empty:r,maxAttempts:n,table:i}=this,s=this.hashSeeds.length,o=Math.floor(Math.random()*s);for(;;){let a=this.getHash(o,sr);if(this.swapPending(i,a),sr===r)return!0;if(++t===n)break;o=(o+Math.floor(Math.random()*(s-1))+1)%s}return!1}allocate(t){this.tableSize=t;let{entryStride:r}=this;this.table=new BigUint64Array(t*r),this.maxAttempts=t,this.clearTable(),this.capacity=t*this.loadFactor,this.mungedEmptyKey=void 0}rehash(t,r){Fr&&console.log("rehash begin"),this.allocate(r),this.updateHashFunctions(this.hashSeeds.length);let{empty:n,entryStride:i}=this;for(let s=0,o=t.length;s<o;s+=i)if(t[s]!==n&&(this.storePending(t,s),!this.tryToInsert()))return Fr&&console.log("rehash failed"),!1;return Fr&&console.log("rehash end"),!0}grow(t){Fr&&console.log(`grow: ${t}`);let r=this.table,{tableSize:n}=this;for(;n<t;)n*=2;for(;;){for(let i=0;i<this.maxRehashAttempts;++i)if(this.rehash(r,n)){Fr&&console.log("grow end");return}n*=2}}insertInternal(){for(++this.generation,sr===this.empty&&this.chooseAnotherEmptyKey(),++this.size>this.capacity&&(this.backupPending(),this.grow(this.tableSize*2),this.restorePending());!this.tryToInsert();)this.backupPending(),this.grow(this.tableSize),this.restorePending()}},is=class extends Ro{add(t){return this.has(t)?!1:(Fr&&console.log(`add: ${t}`),sr=t,this.insertInternal(),!0)}[Symbol.iterator](){return this.keys()}};is.prototype.entryStride=1;var ns=0n,$y=0n,ss=class extends Ro{set(t,r){return this.has(t)?!1:(Fr&&console.log(`add: ${t} -> ${r}`),sr=t,ns=r,this.insertInternal(),!0)}get(t){let r=this.indexOf(t);if(r!==-1)return this.table[r+1]}swapPending(t,r){let n=ns;super.swapPending(t,r),t[r+1]=n}storePending(t,r){super.storePending(t,r),ns=t[r+1]}backupPending(){super.backupPending(),$y=ns}restorePending(){super.restorePending(),ns=$y}[Symbol.iterator](){return this.entries()}*entries(){let{empty:t,entryStride:r}=this,{table:n}=this;for(let i=0,s=n.length;i<s;i+=r){let o=n[i];if(o!==t){let a=n[i+1];yield[o,a]}}}};ss.prototype.entryStride=2;var NA=Object.defineProperty,DA=Object.getOwnPropertyDescriptor,PA=(e,t,r,n)=>{for(var i=n>1?void 0:n?DA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&NA(t,r,i),i};var xu=class extends me{hashTable=new ss;changed=new jt;get value(){return this}static makeWithCounterpart(e){let t=new xu;return t.initializeCounterpart(e),t}set_(e,t){return this.hashTable.set(e,t)}set(e,t){if(this.set_(e,t)){let{rpc:r}=this;r&&r.invoke("Uint64Map.set",{id:this.rpcId,key:e,value:t}),this.changed.dispatch(e,!0)}}has(e){return this.hashTable.has(e)}get(e){return this.hashTable.get(e)}[Symbol.iterator](){return this.hashTable.entries()}delete_(e){return this.hashTable.delete(e)}delete(e){if(this.delete_(e)){let{rpc:t}=this;t&&t.invoke("Uint64Map.delete",{id:this.rpcId,key:e}),this.changed.dispatch(e,!1)}}get size(){return this.hashTable.size}assignFrom(e){this.clear();for(let[t,r]of e)this.set(t,r)}clear(){if(this.hashTable.clear()){let{rpc:e}=this;e&&e.invoke("Uint64Map.clear",{id:this.rpcId}),this.changed.dispatch(null,!1)}}toJSON(){let e={};for(let[t,r]of this.hashTable.entries())e[t.toString()]=r.toString();return e}};xu=PA([j("Uint64Map")],xu);X("Uint64Map.set",function(e){let t=this.get(e.id);t.set_(e.key,e.value)&&t.changed.dispatch()});X("Uint64Map.delete",function(e){let t=this.get(e.id);t.delete_(e.key)&&t.changed.dispatch()});X("Uint64Map.clear",function(e){let t=this.get(e.id);t.hashTable.clear()&&t.changed.dispatch()});var RA=Object.defineProperty,OA=Object.getOwnPropertyDescriptor,UA=(e,t,r,n)=>{for(var i=n>1?void 0:n?OA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&RA(t,r,i),i};var $n=class extends me{hashTable=new is;changed=new jt;get value(){return this}static makeWithCounterpart(e){let t=new $n;return t.initializeCounterpart(e),t}set(e,t){t?this.add(e):this.delete(e)}reserve_(e){return this.hashTable.reserve(e)}reserve(e){if(this.reserve_(e)){let{rpc:t}=this;t&&t.invoke("Uint64Set.reserve",{id:this.rpcId,value:e})}}add_(e){let t=!1;for(let r of e)t=this.hashTable.add(r)||t;return t}add(e){let t=typeof e=="bigint"?[e]:e;if(this.add_(t)){let{rpc:r}=this;r&&r.invoke("Uint64Set.add",{id:this.rpcId,value:t}),this.changed.dispatch(e,!0)}}has(e){return this.hashTable.has(e)}[Symbol.iterator](){return this.hashTable.keys()}keys(){return this.hashTable.keys()}delete_(e){let t=!1;for(let r of e)t=this.hashTable.delete(r)||t;return t}delete(e){let t=typeof e=="bigint"?[e]:e;if(this.delete_(t)){let{rpc:r}=this;r&&r.invoke("Uint64Set.delete",{id:this.rpcId,value:t}),this.changed.dispatch(e,!1)}}get size(){return this.hashTable.size}clear(){if(this.hashTable.clear()){let{rpc:e}=this;e&&e.invoke("Uint64Set.clear",{id:this.rpcId}),this.changed.dispatch(null,!1)}}toJSON(){let e=new Array;for(let t of this.keys())e.push(t.toString());return e.sort(),e}assignFrom(e){this.clear();for(let t of e.keys())this.add(t)}};$n=UA([j("Uint64Set")],$n);X("Uint64Set.reserve",function(e){let t=this.get(e.id);t.reserve_(e.value)&&t.changed.dispatch()});X("Uint64Set.add",function(e){let t=this.get(e.id);t.add_(e.value)&&t.changed.dispatch()});X("Uint64Set.delete",function(e){let t=this.get(e.id);t.delete_(e.value)&&t.changed.dispatch()});X("Uint64Set.clear",function(e){let t=this.get(e.id);t.hashTable.clear()&&t.changed.dispatch()});var Vy=["visibleSegments","segmentEquivalences","temporaryVisibleSegments","temporarySegmentEquivalences","useTemporaryVisibleSegments","useTemporarySegmentEquivalences"];function Oo(e,t,r){e.registerDisposer(t.visibleSegments.changed.add(r)),e.registerDisposer(t.segmentEquivalences.changed.add(r))}function Uo(e,t,r){e.registerDisposer(t.temporaryVisibleSegments.changed.add(r)),e.registerDisposer(t.temporarySegmentEquivalences.changed.add(r)),e.registerDisposer(t.useTemporaryVisibleSegments.changed.add(r)),e.registerDisposer(t.useTemporarySegmentEquivalences.changed.add(r))}function on(e){return e.toString()}function LA(e){return(e&0x8000000000000000n)!==0n}function FA(e){return e.useTemporaryVisibleSegments.value?e.temporaryVisibleSegments:e.visibleSegments}function BA(e){return e.useTemporarySegmentEquivalences.value?e.temporarySegmentEquivalences:e.segmentEquivalences}function xr(e,t){let r=FA(e),n=BA(e),i=n.disjointSets.visibleSegmentEquivalencePolicy.value;for(let s of r.keys())if(i&Lr.NONREPRESENTATIVE_EXCLUDED){let o=n.get(s);t(s,o)}else{if(!n.disjointSets.isMinElement(s))continue;for(let o of n.setElements(s))i&Lr.REPRESENTATIVE_EXCLUDED&&i&Lr.MAX_REPRESENTATIVE&&LA(o)||t(o,s)}}function Su(e,t,r={}){for(let n of Vy)r[n]=e.get(t[n]);return r}var an=e=>class extends e{visibleSegments;selectedSegments;segmentEquivalences;temporaryVisibleSegments;temporarySegmentEquivalences;useTemporaryVisibleSegments;useTemporarySegmentEquivalences;transform;renderScaleTarget;constructor(...r){let[n,i]=r;super(n,i),Su(n,i,this),this.transform=n.get(i.transform),this.renderScaleTarget=n.get(i.renderScaleTarget);let s=()=>{this.chunkManager.scheduleUpdateChunkPriorities()};Uo(this,this,s),Oo(this,this,s),this.registerDisposer(this.transform.changed.add(s)),this.registerDisposer(this.renderScaleTarget.changed.add(s))}};var zA=Object.defineProperty,$A=Object.getOwnPropertyDescriptor,Lo=(e,t,r,n)=>{for(var i=n>1?void 0:n?$A(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&zA(t,r,i),i};var VA=200,jA=60,wu=class extends fe{annotation;freeSystemMemory(){this.annotation=void 0}serialize(t,r){super.serialize(t,r),t.annotation=this.annotation}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=0,super.downloadSucceeded()}},Vn=class{data;typeToOffset;typeToIds;typeToIdMaps;typeToInstanceCounts;typeToSize;serialize(t,r){t.data=this.data,t.typeToOffset=this.typeToOffset,t.typeToIds=this.typeToIds,t.typeToIdMaps=this.typeToIdMaps,t.typeToInstanceCounts=this.typeToInstanceCounts,t.typeToSize=this.typeToSize,r.push(this.data.buffer)}get numBytes(){return this.data.byteLength}};function Yy(e){class t extends e{data;serialize(n,i){super.serialize(n,i);let{data:s}=this;s!==void 0&&(s.serialize(n,i),this.data=void 0)}downloadSucceeded(){let{data:n}=this;this.systemMemoryBytes=this.gpuMemoryBytes=n===void 0?0:n.numBytes,super.downloadSucceeded()}freeSystemMemory(){this.data=void 0}}return t}var Eu=class extends Yy(Fn){},bu=class extends Yy(fe){objectId},jy=class extends Ne{parent=void 0;getChunk(e){let{chunks:t}=this,r=t.get(e);return r===void 0&&(r=this.getNewChunk_(wu),r.initialize(e),this.addChunk(r)),r}download(e,t){return this.parent.downloadMetadata(e,t)}};jy=Lo([j(wy)],jy);var cn=class extends Bn{parent;constructor(t,r){super(t,r),this.parent=t.get(r.parent)}};cn.prototype.chunkConstructor=Eu;var Gy=class extends Ne{parent=void 0;relationshipIndex;getChunk(e){let t=on(e),{chunks:r}=this,n=r.get(t);return n===void 0&&(n=this.getNewChunk_(bu),n.initialize(t),n.objectId=e,this.addChunk(n)),n}download(e,t){return this.parent.downloadSegmentFilteredGeometry(e,this.relationshipIndex,t)}};Gy=Lo([j(Ey)],Gy);var jn=class extends me{references=new Set;chunkManager;metadataChunkSource;segmentFilteredSources;constructor(t,r){super(t,r);let n=this.chunkManager=t.get(r.chunkManager),i=this.metadataChunkSource=this.registerDisposer(t.getRef(r.metadataChunkSource));this.segmentFilteredSources=r.segmentFilteredSource.map((s,o)=>{let a=this.registerDisposer(t.getRef(s));return a.parent=this,a.relationshipIndex=o,a}),i.parent=this,this.registerDisposer(n.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}recomputeChunkPriorities(){let{chunkManager:t,metadataChunkSource:r}=this;for(let n of this.references)t.requestChunk(r.getChunk(n),se.VISIBLE,VA)}add(t){throw new Error("Not implemented")}delete(t){throw new Error("Not implemented")}update(t,r){throw new Error("Not implemented")}};X(by,function(e){let t=this.get(e.id);t.references.add(e.annotation),t.chunkManager.scheduleUpdateChunkPriorities()});X(Iy,function(e){let t=this.get(e.id);t.references.delete(e.annotation),t.chunkManager.scheduleUpdateChunkPriorities()});X(_y,function(e){let t=this.get(e.id),r=e.annotationId,n=e.newAnnotation,i;r===void 0?i=t.add(n).then(s=>({...n,id:s})):n===null?i=t.delete(r).then(()=>null):i=t.update(r,n).then(()=>n),i.then(s=>{t.wasDisposed||this.invoke(mu,{id:t.rpcId,annotationId:r||n.id,newAnnotation:s})},s=>{t.wasDisposed||this.invoke(mu,{id:t.rpcId,annotationId:r||n?.id,error:s.message})})});var qy=class extends Xe(ir){localPosition;renderScaleTarget;constructor(e,t){super(e,t),this.renderScaleTarget=e.get(t.renderScaleTarget),this.localPosition=e.get(t.localPosition);let r=()=>this.chunkManager.scheduleUpdateChunkPriorities();this.registerDisposer(this.localPosition.changed.add(r)),this.registerDisposer(this.renderScaleTarget.changed.add(r)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(t),e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.state={displayDimensionRenderInfo:r.projectionParameters.value.displayDimensionRenderInfo,transformedSources:[]}}recomputeChunkPriorities(){this.chunkManager.registerLayer(this);for(let e of this.attachments.values()){let{view:t}=e,r=t.visibility.value;if(r===Number.NEGATIVE_INFINITY)continue;let n=e.state,{transformedSources:i}=n;if(i.length===0||!On(n,t.projectionParameters.value.displayDimensionRenderInfo))continue;let s=je(r),o=Ge(r),a=t.projectionParameters.value,{chunkManager:c}=this;Ty(a,this.localPosition.value,this.renderScaleTarget.value,i[0],()=>{},(l,u)=>{let f=l.source.getChunk(l.curPositionInChunks);++this.numVisibleChunksNeeded,f.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable,c.requestChunk(f,s,o+0+ts*u)})}}};qy=Lo([j(Cy)],qy);X(Ay,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);n.state.transformedSources=sn(this,e.sources,r),n.state.displayDimensionRenderInfo=e.displayDimensionRenderInfo,r.chunkManager.scheduleUpdateChunkPriorities()});var Ky=class extends Et(Xe(Dr)){source;segmentationStates;constructor(e,t){super(e,t),this.source=e.get(t.source),this.segmentationStates=new Gt(this.getSegmentationState(t.segmentationStates));let r=()=>this.chunkManager.scheduleUpdateChunkPriorities();this.registerDisposer(rm((n,i)=>{if(i!==void 0){for(let s of i)s!=null&&(Oo(n,s,r),Uo(n,s,r));r()}},this.segmentationStates)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}recomputeChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;let{segmentationStates:{value:t},source:{segmentFilteredSources:r}}=this;if(t===void 0)return;let{chunkManager:n}=this;n.registerLayer(this);let i=t.length;for(let s=0;s<i;++s){let o=t[s];if(o==null)continue;let a=je(e),c=Ge(e),l=r[s];xr(o,u=>{let f=l.getChunk(u);++this.numVisibleChunksNeeded,f.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable,n.requestChunk(f,a,c+jA)})}}getSegmentationState(e){if(e!==void 0)return e.map(t=>t==null?t:Su(this.rpc,t))}};Ky=Lo([j(My)],Ky);X(ky,function(e){let t=this.get(e.id);t.segmentationStates.value=t.getSegmentationState(e.segmentationStates)});var bt=class e extends Error{url;status;statusText;response;constructor(t,r,n,i,s){let o=`Fetching ${JSON.stringify(t)} resulted in HTTP error ${r}`;n&&(o+=`: ${n}`),o+=".",super(o,s),this.name="HttpError",this.message=o,this.url=t,this.status=r,this.statusText=n,i&&(this.response=i)}static fromResponse(t){return new e(t.url,t.status,t.statusText,t)}static fromRequestError(t,r){if(r instanceof TypeError){let n;return typeof t=="string"?n=t:n=t.url,new e(n,0,"Network or CORS error",void 0,{cause:r})}return r}},GA=32,qA=500,KA=1e4;function Iu(e){return Math.min(2**e*qA,KA/2)*(1+Math.random())}async function Re(e,t){for(let r=0;;){t?.signal?.throwIfAborted();let n;try{n=await fetch(e,t)}catch(i){throw bt.fromRequestError(e,i)}if(!n.ok){let{status:i}=n;if((i===429||i===503||i===504)&&++r!==GA){await new Promise(s=>setTimeout(s,Iu(r-1)));continue}throw bt.fromResponse(n)}return n}}function Jy(e){return e instanceof bt?e.status===0||e.status===403||e.status===404:!1}var YA=3;async function ln(e,t,r,n,i){let s;for(let o=0;;){r.signal?.throwIfAborted(),o>1&&await new Promise(a=>setTimeout(a,Iu(o-2))),s=await e.get(s,{signal:r.signal??void 0,progressListener:r.progressListener});try{return await Re(typeof t=="function"?t(s.credentials):t,n(s.credentials,r))}catch(a){if(a instanceof bt&&i(a,s.credentials)==="refresh"){if(++o===YA)throw a;continue}throw a}}}function Hy(e,t,r){return(n,i={})=>ln(e,n,i,t,r)}async function Fo(e,t,r){return Re(t,r).catch(n=>{if(n.status!==500&&n.status!==401&&n.status!==403&&n.status!==504)throw n;return ln(e,t,r,i=>{let s=new Headers(r.headers);return s.set("Authorization",`Bearer ${i}`),{...r,headers:s}},i=>{let{status:s}=i;if(s===403||s===401)return"refresh";throw i})})}var _u=class{baseUrl;collection;experiment;channel;resolution},Bo=class extends _u{encoding;window;static RPC_ID="boss/VolumeChunkSource";static stringify(t){return`boss:volume:${t.baseUrl}/${t.collection}/${t.experiment}/${t.channel}/${t.resolution}/${t.encoding}`}},zo=class{baseUrl;static RPC_ID="boss/MeshChunkSource";static stringify(t){return`boss:mesh:${t.baseUrl}`}};var Wy="mesh/MeshLayer",Xy="mesh/MultiscaleMeshLayer",Zy="mesh/FragmentSource",Qy="mesh/MultiscaleFragmentSource",un=(e=>(e[e.float32=0]="float32",e[e.uint10=1]="uint10",e[e.uint16=2]="uint16",e))(un||{});function Cu(e,t,r){return e&1|t<<1&2|r<<2&4}function t0(e,t,r,n){let i=Math.max(t,r,n),s=0,o=0,a=0,c=0;for(let l=0;l<i;++l){if(l<t){let u=Number(e>>BigInt(s++)&BigInt(1));o|=u<<l}if(l<r){let u=Number(e>>BigInt(s++)&BigInt(1));a|=u<<l}if(l<n){let u=Number(e>>BigInt(s++)&BigInt(1));c|=u<<l}}return Uint32Array.of(o,a,c)}function $o(e,t,r,n,i,s){let o=Math.max(e,t,r),a=0,c=0n;function l(u){c|=BigInt(u)<<BigInt(a++)}for(let u=0;u<o;++u)u<e&&l(n>>u&1),u<t&&l(i>>u&1),u<r&&l(s>>u&1);return c}function r0(e,t){let r=0n,n=0,i=e.length;function s(o){r|=BigInt(o&1)<<BigInt(n++)}for(let o=0;o<32;++o)for(let a=0;a<i;++a)t[a]-1>>>o&&s(e[a]>>>o);return r}function e0(e,t){return e<t&&e<(e^t)}function Gn(e,t,r,n,i,s){let o=r,a=s;return e0(o^a,t^i)&&(o=t,a=i),e0(o^a,e^n)&&(o=e,a=n),o<a}function n0(e,t,r,n,i,s,o){let{octree:a,lodScales:c,chunkGridSpatialOrigin:l,chunkShape:u}=e,f=c.length-1,h=t[0],p=t[4],d=t[8],m=t[1],g=t[5],y=t[9],I=t[3],_=t[7],E=t[11],b=t[15],C=I>0?0:1,v=_>0?0:1,w=E>0?0:1,x=r[4*4],k=r[4*4+1],M=r[4*4+2],P=r[4*4+3];function F(ze,at,gt){return I*ze+_*at+E*gt+b}function S(ze,at,gt,yt,vt,dr){return F(ze+C*(yt-ze),at+v*(vt-at),gt+w*(dr-gt))}let O=F(-P*x,-P*k,-P*M),R=e.clipLowerBound[0],N=e.clipLowerBound[1],U=e.clipLowerBound[2],L=e.clipUpperBound[0],G=e.clipUpperBound[1],V=e.clipUpperBound[2],Y=Math.sqrt((h*i)**2+(m*s)**2),J=Math.sqrt((p*i)**2+(g*s)**2),de=Math.sqrt((d*i)**2+(y*s)**2),ie=Math.max(Y,J,de);function Ce(ze,at,gt){let yt=1<<ze,vt=at*5,dr=a[vt],kr=a[vt+1],ki=a[vt+2],Cw=a[vt+3],qh=a[vt+4],Ti=dr*yt*u[0]+l[0],Ni=kr*yt*u[1]+l[1],Di=ki*yt*u[2]+l[2],Fs=Ti+yt*u[0],Bs=Ni+yt*u[1],zs=Di+yt*u[2];if(Ti=Math.max(Ti,R),Ni=Math.max(Ni,N),Di=Math.max(Di,U),Fs=Math.min(Fs,L),Bs=Math.min(Bs,G),zs=Math.min(zs,V),po(Ti,Ni,Di,Fs,Bs,zs,r)){let $c=Math.max(O,S(Ti,Ni,Di,Fs,Bs,zs))/ie;if(gt===0||$c*n<gt){let En=c[ze];if(En!==0&&o(ze,at,En/$c,qh>>>31),ze>0&&(En===0||$c*n<En)){let Aw=En===0?gt:En,Mw=(qh&2147483647)>>>0;for(let Vc=Cw;Vc<Mw;++Vc)Ce(ze-1,Vc,Aw)}}}}Ce(f,a.length/5-1,0)}var Vo=!1;function JA(e){let t=0;for(let r=0,n=e.length;r<n;r+=3){let i=e[r],s=e[r+1],o=e[r+2],a;i>s&&(a=i,i=s,s=a),s>o&&(a=s,s=o,o=a),i>s&&(a=i,i=s,s=a),e[r]=i,e[r+1]=s,e[r+2]=o,o>t&&(t=o)}return t}var Au=0;function HA(e,t,r,n,i,s){let o=t-1>>>0,a=(i&o)>>>0;for(let c=0;;++c){let l=e[a];if(l===n)return e[a]=r,r;if(s(l))return l;++Au,a=(a+c+1&o)>>>0}}function WA(e,t){return zn(zn(0,e),t)}var XA=1053042;function ZA(e){return XA>>>e*3&7}function jo(e){return(e>>>2)*3}function Go(e){return e&3}function Mu(e){return e>>>1}function ku(e){return 1+(e+1>>>1)}function i0(e){return 2-e}function QA(e){return 2**Math.ceil(Math.log2(e))*4}function eM(e,t,r){let n=t.length/3,i=r.length,s=4294967295;e.fill(s),r.fill(s);for(let o=0;o<n;++o){let a=o*3;for(let c=0;c<3;++c){let l=t[a+Mu(c)],u=t[a+ku(c)],f=o<<2|c,h=HA(r,i,f,s,WA(l,u),p=>{let d=jo(p),m=Go(p),g=t[d+Mu(m)],y=t[d+ku(m)];return l===g&&u===y});if(h!==f){let p=jo(h),d=Go(h);e[p+d]=f,e[a+c]=h}}}return e}function tM(e,t,r,n){let i=-1>>>32-8*r.BYTES_PER_ELEMENT,o=e.length/3,a=4294967295;e:for(let c=0;c<o;++c){let l=c*3;if(e[l]!==i){for(let u=0;u<3;++u){let f=t[l+u];if(f===a)continue;let h=jo(f);if(e[h]===i)continue;let p=Go(f);r[n++]=e[l+i0(u)],r[n++]=e[l+Mu(u)],r[n++]=e[l+ku(u)];let d=p;for(;;){if(e[l]=i,l=h,r[n++]=e[l+i0(d&3)],d=ZA(d),f=t[l+(d&3)],f===a||e[h=jo(f)]===i){r[n++]=i,e[l]=i;continue e}d=Go(f)|d&4}}r[n++]=e[l],r[n++]=e[l+1],r[n++]=e[l+2],e[l]=i,r[n++]=i}}return n}function s0(e,t){if(e.length===0)return e;Au=0,t===void 0&&(t=Uint32Array.of(0,e.length));let r=0,n=0,i=0,s=0,o=0,a=JA(e),c=e.length/3*4,l=a>=65535?new Uint32Array(c):new Uint16Array(c),u=0,f=0,h=t.length-1;for(let y=0;y<h;++y)f=Math.max(f,t[y+1]-t[y]);let p=new Uint32Array(f),d=new Uint32Array(QA(f)),m=t[0];for(let y=0;y<h;++y){t[y]=u;let I=t[y+1],_=e.subarray(m,I);Vo&&(i=Date.now()),eM(p,_,d),Vo&&(s=Date.now()),u=tM(_,p,l,u),Vo&&(o=Date.now(),r+=s-i,n+=o-s),m=I}--u,t[h]=u;let g=new l.constructor(u);return g.set(l.subarray(0,u)),Vo&&console.log(`reduced from ${e.byteLength}(${e.BYTES_PER_ELEMENT}) -> ${g.byteLength}(${g.BYTES_PER_ELEMENT}): adj=${r}, emit=${n}, ${Au}/${e.length} collisions`),g}var oe=(e=>(e[e.LITTLE=0]="LITTLE",e[e.BIG=1]="BIG",e))(oe||{});function rM(){let e=Uint16Array.of(4386);return new Uint8Array(e.buffer)[0]===17?1:0}var Sr=rM();function o0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);for(let r=0,n=t.length;r<n;r+=2){let i=t[r];t[r]=t[r+1],t[r+1]=i}}function a0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);for(let r=0,n=t.length;r<n;r+=4){let i=t[r];t[r]=t[r+3],t[r+3]=i,i=t[r+1],t[r+1]=t[r+2],t[r+2]=i}}function c0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);for(let r=0,n=t.length;r<n;r+=8){let i=t[r];t[r]=t[r+7],t[r+7]=i,i=t[r+1],t[r+1]=t[r+6],t[r+6]=i,i=t[r+2],t[r+2]=t[r+5],t[r+5]=i,i=t[r+3],t[r+3]=t[r+4],t[r+4]=i}}function l0(e,t,r=Sr){t!==r&&o0(e)}function It(e,t,r=Sr){t!==r&&a0(e)}function u0(e,t,r=Sr){t!==r&&c0(e)}function qn(e,t,r,n=Sr){if(!(t===n||r===1))switch(r){case 2:o0(e);break;case 4:a0(e);break;case 8:c0(e);break}}var nM=Object.defineProperty,iM=Object.getOwnPropertyDescriptor,Ko=(e,t,r,n)=>{for(var i=n>1?void 0:n?iM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&nM(t,r,i),i};var g0=100,y0=50,sM=!1,Tu=class extends fe{objectId=0n;fragmentIds;initializeManifestChunk(t,r){super.initialize(t),this.objectId=r}freeSystemMemory(){this.fragmentIds=null}serialize(t,r){super.serialize(t,r),t.fragmentIds=this.fragmentIds}downloadSucceeded(){this.systemMemoryBytes=100,this.gpuMemoryBytes=0,super.downloadSucceeded(),this.priorityTier<se.RECENT&&this.source.chunkManager.scheduleUpdateChunkPriorities()}toString(){return this.objectId.toString()}};function v0(e,t,r){let{vertexPositions:n,indices:i,vertexNormals:s,strips:o}=e;t.vertexPositions=n,t.indices=i,t.strips=o,t.vertexNormals=s;let a=n.buffer;r.push(a);let c=i.buffer;c!==a&&r.push(c),r.push(s.buffer)}function x0(e){let{vertexPositions:t,indices:r,vertexNormals:n}=e;return t.byteLength+r.byteLength+n.byteLength}var Nu=class extends fe{manifestChunk=null;fragmentId=null;meshData=null;initializeFragmentChunk(t,r,n){super.initialize(t),this.manifestChunk=r,this.fragmentId=n}freeSystemMemory(){this.manifestChunk=null,this.meshData=null,this.fragmentId=null}serialize(t,r){super.serialize(t,r),v0(this.meshData,t,r),this.meshData=null}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=x0(this.meshData),super.downloadSucceeded()}};function Yo(e,t,r){Ee(t),e.fragmentIds=re(t,r,Yt)}function Ru(e,t){let r=B.create(),n=B.create(),i=B.create(),s=new Float32Array(e.length),o=t.length;for(let c=0;c<o;c+=3){let l=t[c]*3,u=t[c+1]*3,f=t[c+2]*3;for(let h=0;h<3;++h)n[h]=e[u+h]-e[l+h],i[h]=e[f+h]-e[u+h];B.cross(r,n,i),B.normalize(r,r);for(let h=0;h<3;++h){let d=t[c+h]*3;for(let m=0;m<3;++m)s[d+m]+=r[m]}}let a=s.length;for(let c=0;c<a;c+=3){let l=s.subarray(c,c+3);B.normalize(l,l)}return s}function qo(e){return Math.min(Math.max(-127,e*127+.5),127)>>>0}function f0(e){return e<0?-1:1}function oM(e,t){let r=t.length,n=0;for(let i=0;i<r;i+=3){let s=t[i],o=t[i+1],a=t[i+2],c=1/(Math.abs(s)+Math.abs(o)+Math.abs(a));a<0?(e[n]=qo((1-Math.abs(o*c))*f0(s)),e[n+1]=qo((1-Math.abs(s*c))*f0(o))):(e[n]=qo(s*c),e[n+1]=qo(o*c)),n+=2}}function Ou(e,t,r,n,i,s,o){let a=new Float32Array(t,n,i*3);It(a,r),s===void 0&&(s=n+12*i);let c;o!==void 0&&(c=o*e);let l=c===void 0?new Uint32Array(t,s):new Uint32Array(t,s,c);if(l.length%e!==0)throw new Error(`Number of indices is not a multiple of ${e}: ${l.length}.`);return It(l,r),{vertexPositions:a,indices:l}}function Yn(e,t,r,n,i,s){return Ou(3,e,t,r,n,i,s)}var Nt=class extends Ne{fragmentSource;constructor(t,r){super(t,r);let n=this.fragmentSource=this.registerDisposer(t.getRef(r.fragmentSource));n.meshSource=this}getChunk(t){let r=on(t),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(Tu),n.initializeManifestChunk(r,t),this.addChunk(n)),n}getFragmentKey(t,r){return{key:`${t}/${r}`,fragmentId:r}}getFragmentChunk(t,r){let n=this.fragmentSource,{key:i,fragmentId:s}=this.getFragmentKey(t.key,r),o=n.chunks.get(i);return o===void 0&&(o=n.getNewChunk_(Nu),o.initializeFragmentChunk(i,t,s),n.addChunk(o)),o}},h0=class extends Ne{meshSource=null;download(e,t){return this.meshSource.downloadFragment(e,t)}};h0=Ko([j(Zy)],h0);var p0=class extends an(Et(Xe(rs))){source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}attach(e){let t=()=>{this.chunkManager.scheduleUpdateChunkPriorities()},{view:r}=e;e.registerDisposer(r.visibility.changed.add(t)),e.registerDisposer(t),t()}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;this.chunkManager.registerLayer(this);let t=je(e),r=Ge(e),{source:n,chunkManager:i}=this;xr(this,s=>{let o=n.getChunk(s);++this.numVisibleChunksNeeded,i.requestChunk(o,t,r+g0);let a=o.state;if(a===z.SYSTEM_MEMORY_WORKER||a===z.SYSTEM_MEMORY||a===z.GPU_MEMORY){++this.numVisibleChunksAvailable;for(let c of o.fragmentIds){let l=n.getFragmentChunk(o,c);++this.numVisibleChunksNeeded,i.requestChunk(l,t,r+y0),l.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable}}})}};p0=Ko([j(Wy)],p0);var Du=class extends fe{objectId=0n;manifest;initializeManifestChunk(t,r){super.initialize(t),this.objectId=r}freeSystemMemory(){this.manifest=void 0}serialize(t,r){super.serialize(t,r),t.manifest=this.manifest}downloadSucceeded(){this.systemMemoryBytes=this.manifest.octree.byteLength,this.gpuMemoryBytes=0,super.downloadSucceeded(),this.priorityTier<se.RECENT&&this.source.chunkManager.scheduleUpdateChunkPriorities()}toString(){return this.objectId.toString()}},Pu=class extends fe{subChunkOffsets=null;meshData=null;lod=0;chunkIndex=0;manifestChunk=null;freeSystemMemory(){this.meshData=this.subChunkOffsets=null}serialize(t,r){super.serialize(t,r),v0(this.meshData,t,r);let{subChunkOffsets:n}=this;t.subChunkOffsets=n,r.push(n.buffer),this.meshData=this.subChunkOffsets=null}downloadSucceeded(){let{subChunkOffsets:t}=this;this.systemMemoryBytes=this.gpuMemoryBytes=x0(this.meshData),this.systemMemoryBytes+=t.byteLength,super.downloadSucceeded()}},Kn=class extends Ne{fragmentSource;format;constructor(t,r){super(t,r);let n=this.fragmentSource=this.registerDisposer(t.getRef(r.fragmentSource));this.format=r.format,n.meshSource=this}getChunk(t){let r=on(t),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(Du),n.initializeManifestChunk(r,t),this.addChunk(n)),n}getFragmentChunk(t,r,n){let i=`${t.key}/${r}:${n}`,s=this.fragmentSource,o=s.chunks.get(i);return o===void 0&&(o=s.getNewChunk_(Pu),o.initialize(i),o.lod=r,o.chunkIndex=n,o.manifestChunk=t,s.addChunk(o)),o}},d0=class extends Ne{meshSource=null;download(e,t){return this.meshSource.downloadFragment(e,t)}};d0=Ko([j(Qy)],d0);var aM=ge.create(),m0=class extends an(Et(Xe(rs))){source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.registerDisposer(t),t()}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;let{transform:{value:t}}=this;if(t.error!==void 0)return;let r=new Array;this.chunkManager.registerLayer(this);{let s=je(e),o=Ge(e),{source:a,chunkManager:c}=this;xr(this,l=>{let u=a.getChunk(l);++this.numVisibleChunksNeeded,c.requestChunk(u,s,o+g0);let f=u.state;(f===z.SYSTEM_MEMORY_WORKER||f===z.SYSTEM_MEMORY||f===z.GPU_MEMORY)&&(r.push(u),++this.numVisibleChunksAvailable)})}if(r.length===0)return;let{source:n,chunkManager:i}=this;for(let{view:s}of this.attachments.values()){let o=s.visibility.value;if(o===Number.NEGATIVE_INFINITY)continue;let a=je(o),c=Ge(o),l=s.projectionParameters.value,u=aM;try{Jg(u,l.displayDimensionRenderInfo,t)}catch{continue}ge.multiply(u,l.viewProjectionMat,u);let f=ho(new Float32Array(24),u),h=this.renderScaleTarget.value;for(let p of r){let d=p.manifest.lodScales.length-1;n0(p.manifest,u,f,h,l.width,l.height,(m,g,y,I)=>{if(I)return;let _=n.getFragmentChunk(p,m,g);++this.numVisibleChunksNeeded,i.requestChunk(_,a,c+y0-d+m),_.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable})}}}};m0=Ko([j(Xy)],m0);function S0(e,t){let r=Ru(e.vertexPositions,e.indices),n=new Uint8Array(r.length/3*2);oM(n,r);let i,s;sM?(i=s0(e.indices,e.subChunkOffsets),s=!0):(e.indices.BYTES_PER_ELEMENT===4&&e.vertexPositions.length/3<65535?(i=new Uint16Array(e.indices.length),i.set(e.indices)):i=e.indices,s=!1);let o;if(t===un.uint10){let a=e.vertexPositions,c=a.length/3;o=new Uint32Array(c);for(let l=0,u=0;u<c;l+=3,++u)o[u]=a[l]&1023|(a[l+1]&1023)<<10|(a[l+2]&1023)<<20}else if(t===un.uint16){let a=e.vertexPositions;a.BYTES_PER_ELEMENT===2?o=a:(o=new Uint16Array(a.length),o.set(a))}else o=e.vertexPositions;return{vertexPositions:o,vertexNormals:n,indices:i,strips:s}}function or(e,t,r=un.float32){e.meshData=S0(t,r)}function Jo(e,t,r){e.meshData=S0(t,r),e.subChunkOffsets=t.subChunkOffsets}function Ho(e,t,r){let n=r;for(let i=0;i<3;++i)e[n*5+i]=e[t*5+i]>>>1;e[n*5+3]=t;for(let i=t+1;i<r;++i){let s=e[i*5]>>>1,o=e[i*5+1]>>>1,a=e[i*5+2]>>>1;(s!==e[n*5]||o!==e[n*5+1]||a!==e[n*5+2])&&(e[n*5+4]=i,++n,e[n*5]=s,e[n*5+1]=o,e[n*5+2]=a,e[n*5+3]=i)}return e[n*5+4]=r,++n,n}function w0(e,t,r,n){let i=t;for(let s=r;s<n;++s){let o=e[s*5],a=e[s*5+1],c=e[s*5+2];for(;i<r;){let l=e[i*5]>>>1,u=e[i*5+1]>>>1,f=e[i*5+2]>>>1;if(!Gn(l,u,f,o,a,c))break;++i}for(e[s*5+3]=i;i<r;){let l=e[i*5]>>>1,u=e[i*5+1]>>>1,f=e[i*5+2]>>>1;if(l!==o||u!==a||f!==c)break;++i}e[s*5+4]+=i}}function De(e){return{id:e}}var E0=De("encodeCompressedSegmentationUint32"),b0=De("encodeCompressedSegmentationUint64");var Uu=0,Lu=[],Wo=new Map,Jn=new Map,cM=typeof navigator.hardwareConcurrency>"u"?4:Math.min(12,navigator.hardwareConcurrency),lM=0;function I0(e){for(let[t,r]of Wo){Wo.delete(t),r.cleanup?.(),e.postMessage(r.msg,r.transfer);return}Lu.push(e)}function uM(){++Uu;let e=new Worker(new URL("../async_computation.bundle.js",import.meta.url),{type:"module"}),t=!1;e.onmessage=r=>{if(!t){t=!0,I0(e);return}let{id:n,value:i,error:s}=r.data;I0(e);let o=Jn.get(n);Jn.delete(n),o!==void 0&&(s!==void 0?o.reject(s):o.resolve(i))}}function ue(e,t,r,...n){let i=lM++,s={t:e.id,id:i,args:n};t?.throwIfAborted();let o=new Promise((c,l)=>{Jn.set(i,{resolve:c,reject:l})});if(Lu.length!==0)Lu.pop().postMessage(s,r);else{let c;if(t!==void 0){let l=function(){Wo.delete(i);let u=Jn.get(i);Jn.delete(i),u.reject(t.reason)};var a=l;t.addEventListener("abort",l,{once:!0}),c=()=>{t.removeEventListener("abort",l)}}Wo.set(i,{msg:s,transfer:r,cleanup:c}),Jn.size>Uu&&Uu<cM&&uM()}return o}async function ut(e,t,r){let{spec:n}=e.source;if(n.compressedSegmentationBlockSize!==void 0){let{dataType:i}=n,s=e.chunkDataSize,o=[s[0],s[1],s[2],s[3]||1];switch(i){case W.UINT32:e.data=await ue(E0,t,[r.buffer],r,o,n.compressedSegmentationBlockSize);break;case W.UINT64:e.data=await ue(b0,t,[r.buffer],r,o,n.compressedSegmentationBlockSize);break;default:throw new Error(`Unsupported data type for compressed segmentation: ${W[i]}`)}}else e.data=r}function _0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);return t.length>=3&&t[0]===31&&t[1]===139&&t[2]===8}async function wr(e,t,r){try{let n=Fu(e instanceof Response?e:new Response(e),t,r);return await new Response(n).arrayBuffer()}catch{throw r?.throwIfAborted(),new Error(`Failed to decode ${t}`)}}function Fu(e,t,r){return e.body.pipeThrough(new DecompressionStream(t),{signal:r})}var Er=new Map;Er.set("|u1",{endianness:oe.LITTLE,dataType:W.UINT8});Er.set("|i1",{endianness:oe.LITTLE,dataType:W.INT8});for(let[e,t]of[["<",oe.LITTLE],[">",oe.BIG]]){for(let r of["u","i"])Er.set(`${e}${r}8`,{endianness:t,dataType:W.UINT64});Er.set(`${e}u2`,{endianness:t,dataType:W.UINT16}),Er.set(`${e}i2`,{endianness:t,dataType:W.INT16}),Er.set(`${e}u4`,{endianness:t,dataType:W.UINT32}),Er.set(`${e}i4`,{endianness:t,dataType:W.INT32}),Er.set(`${e}f4`,{endianness:t,dataType:W.FLOAT32})}function Bu(e){let t=Er.get(e);if(t===void 0)throw new Error(`Unsupported numpy data type: ${JSON.stringify(e)}`);return t}var zu=class{constructor(t,r,n,i){this.data=t,this.shape=r,this.dataType=n,this.fortranOrder=i}};function C0(e){if(e[0]!==147||e[1]!==78||e[2]!==85||e[3]!==77||e[4]!==80||e[5]!==89)throw new Error("Data does not match npy format.");let t=e[6],r=e[7];if(t!==1||r!==0)throw new Error(`Unsupported npy version ${t}.${r}`);let i=new DataView(e.buffer,e.byteOffset,e.byteLength).getUint16(8,!0),s=new TextDecoder("utf-8").decode(e.subarray(10,i+10)),o,a=i+10;try{o=ug(s)}catch(g){throw new Error(`Failed to parse npy header: ${g}`)}let c=o.descr,l=o.shape,u=1;if(!Array.isArray(l))throw new Error("Invalid shape ${JSON.stringify(shape)}");for(let g of l){if(typeof g!="number")throw new Error("Invalid shape ${JSON.stringify(shape)}");u*=g}let{dataType:f,endianness:h}=Bu(c),p=lt[f],d=uu[f];if(p*u+a!==e.byteLength)throw new Error("Expected length does not match length of data");let m=new d(e.buffer,e.byteOffset+a,u);return qn(m,h,p),new zu(m,l,f,o.fortran_order===!0)}async function A0(e,t,r){let n=C0(new Uint8Array(await wr(r,"deflate"))),i=e.chunkDataSize,s=e.source,{shape:o}=n;if(o.length!==3||o[0]!==i[2]||o[1]!==i[1]||o[2]!==i[0])throw new Error(`Shape ${JSON.stringify(o)} does not match chunkDataSize ${tr(i)}`);let a=n.dataType,{spec:c}=s;if(a!==c.dataType)throw new Error(`Data type ${W[a]} does not match expected data type ${W[c.dataType]}`);await ut(e,t,n.data)}var Hn=De("decodeJpeg");async function Br(e,t,r){let n=e.chunkDataSize,{uint8Array:i}=await ue(Hn,t,[r],new Uint8Array(r),void 0,void 0,n[0]*n[1]*n[2],n[3]||1,!1);await ut(e,t,i)}var $u=class extends Fn{source=null;data;chunkDataSize;initializeVolumeChunk(t,r){super.initializeVolumeChunk(t,r),this.chunkDataSize=null,this.data=null}serialize(t,r){super.serialize(t,r);let n=this.chunkDataSize;n!==this.source.spec.chunkDataSize&&(t.chunkDataSize=n);let i=t.data=this.data;i!==null&&r.push(i.buffer),this.data=null}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=this.data?.byteLength??0,super.downloadSucceeded()}freeSystemMemory(){this.data=null}};function Vu(e,t){let{spec:r,tempChunkDataSize:n,tempChunkPosition:i}=e,{upperVoxelBound:s,rank:o,baseVoxelOffset:a}=r,c=r.chunkDataSize,l=n,u=zg(i,t.chunkGridPosition,c),f=!1;for(let h=0;h<o;++h){let p=Math.min(s[h],u[h]+c[h]);(l[h]=p-u[h])!==c[h]&&(f=!0)}return au(u,u,a),f?t.chunkDataSize=Uint32Array.from(l):t.chunkDataSize=c,u}var Ae=class extends Bn{tempChunkDataSize;tempChunkPosition;constructor(t,r){super(t,r);let n=this.spec.rank;this.tempChunkDataSize=new Uint32Array(n),this.tempChunkPosition=new Float32Array(n)}computeChunkBounds(t){return Vu(this,t)}};Ae.prototype.chunkConstructor=$u;var fM=Object.defineProperty,hM=Object.getOwnPropertyDescriptor,T0=(e,t,r,n)=>{for(var i=n>1?void 0:n?hM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&fM(t,r,i),i};var ju=new Map;ju.set("npz",A0);ju.set("jpeg",Br);var Gu=new Map;Gu.set("npz","application/npygz");Gu.set("jpeg","image/jpeg");function N0(e,t){return ye(Nn()(e),t)}var M0=class extends N0(Ae,Bo){chunkDecoder=ju.get(this.parameters.encoding);async download(e,t){let{parameters:r}=this,n=`${r.baseUrl}/latest/cutout/${r.collection}/${r.experiment}/${r.channel}/${r.resolution}`;{let s=this.computeChunkBounds(e),o=e.chunkDataSize;for(let a=0;a<3;++a)n+=`/${s[a]}:${s[a]+o[a]}`}n+="/",r.window!==void 0&&(n+=`?window=${r.window[0]},${r.window[1]}`);let i=await Fo(this.credentialsProvider,n,{signal:t,headers:{Accept:Gu.get(r.encoding)}});await this.chunkDecoder(e,t,await i.arrayBuffer())}};M0=T0([j()],M0);function pM(e,t){return Yo(e,t,"fragments")}function dM(e,t){let n=new DataView(t).getUint32(0,!0);or(e,Yn(t,oe.LITTLE,4,n))}var k0=class extends N0(Nt,zo){download(e,t){let{parameters:r}=this;return Fo(this.credentialsProvider,`${r.baseUrl}${e.objectId}`,{signal:t}).then(n=>n.arrayBuffer()).then(n=>pM(e,n))}downloadFragment(e,t){let{parameters:r}=this;return Fo(this.credentialsProvider,`${r.baseUrl}${e.fragmentId}`,{signal:t}).then(n=>n.arrayBuffer()).then(n=>dM(e,n))}};k0=T0([j()],k0);var Xo=new Float32Array(1);function D0(e){Xo[0]=e,e=Xo[0];for(let t=1;t<21;++t){let r=e.toPrecision(t);if(Xo[0]=parseFloat(r),Xo[0]===e)return r}return e.toString()}function P0(e){return("0"+e.toString(16)).slice(-2)}function mM(e){let t=/^rgba\(([0-9]+), ([0-9]+), ([0-9]+), (0(?:\.[0-9]+)?)\)$/;{let n=e.match(t);if(n!==null)return[parseInt(n[1],10),parseInt(n[2],10),parseInt(n[3],10),parseFloat(n[4])]}let r=/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/;{let n=e.match(r);if(n!==null)return[parseInt(n[1],16),parseInt(n[2],16),parseInt(n[3],16),1]}throw new Error(`Invalid serialized color: ${JSON.stringify(e)}.`)}function qu(e){try{if(typeof e!="string")throw new Error(`Expected string, but received ${JSON.stringify(e)}.`);let t=document.createElement("canvas").getContext("2d");t.fillStyle=e;let r=mM(t.fillStyle);return gr.fromValues(r[0]/255,r[1]/255,r[2]/255,r[3])}catch(t){throw new Error(`Failed to parse color specification: ${t.message}`)}}function R0(e){return qu(e).subarray(0,3)}function Ku(e){let t=e[3]===void 0?3:4,r=0;for(let n=0;n<t;n++)r=(r<<8>>>0)+Math.min(255,Math.max(0,Math.round(e[t-1-n]*255)));return r}function O0(e){return B.fromValues((e>>>0&255)/255,(e>>>8&255)/255,(e>>>16&255)/255)}function U0(e){return gr.fromValues((e>>>0&255)/255,(e>>>8&255)/255,(e>>>16&255)/255,(e>>>24&255)/255)}function Yu(e){if(e[3]===void 0||e[3]===1){let r="#";for(let n=0;n<3;++n)r+=P0(Math.min(255,Math.max(0,Math.round(e[n]*255))));return r}let t="rgba(";for(let r=0;r<3;++r)r!==0&&(t+=", "),t+=Math.min(255,Math.max(0,Math.round(e[r]*255)));return t+=`, ${D0(e[3])})`,t}var ZB=2**-1074,gM=new Float64Array(1),QB=new Uint32Array(gM.buffer);var fz={[W.UINT8]:[0,255],[W.INT8]:[-128,127],[W.UINT16]:[0,65535],[W.INT16]:[-32768,32767],[W.UINT32]:[0,4294967295],[W.INT32]:[-2147483648,2147483647],[W.UINT64]:[0n,0xffffffffffffffffn],[W.FLOAT32]:[0,1]};var Oe=(e=>(e[e.POINT=0]="POINT",e[e.LINE=1]="LINE",e[e.AXIS_ALIGNED_BOUNDING_BOX=2]="AXIS_ALIGNED_BOUNDING_BOX",e[e.ELLIPSOID=3]="ELLIPSOID",e[e.POLYLINE=4]="POLYLINE",e))(Oe||{}),br=[0,1,2,3,4];var bz={float32:W.FLOAT32,uint32:W.UINT32,int32:W.INT32,uint16:W.UINT16,int16:W.INT16,uint8:W.UINT8,int8:W.INT8,rgb:void 0,rgba:void 0},Hu={rgb:{serializedBytes(){return 3},alignment(){return 1},serializeCode(e,t){return`dv.setUint16(${t}, ${e}, true);dv.setUint8(${t} + 2, ${e} >>> 16);`},deserializeCode(e,t){return`${e} = dv.getUint16(${t}, true) | (dv.getUint8(${t} + 2) << 16);`},deserializeJson(e){return Ku(R0(e))},serializeJson(e){return Yu(O0(e))}},rgba:{serializedBytes(){return 4},alignment(){return 1},serializeCode(e,t){return`dv.setUint32(${t}, ${e}, true);`},deserializeCode(e,t){return`${e} = dv.getUint32(${t}, true);`},deserializeJson(e){return Ku(qu(e))},serializeJson(e){return Yu(U0(e))}},float32:{serializedBytes(){return 4},alignment(){return 4},serializeCode(e,t){return`dv.setFloat32(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getFloat32(${t}, isLittleEndian);`},deserializeJson(e){return go(e)},serializeJson(e){return e}},uint32:{serializedBytes(){return 4},alignment(){return 4},serializeCode(e,t){return`dv.setUint32(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getUint32(${t}, isLittleEndian);`},deserializeJson(e){return Tr(e)},serializeJson(e){return e}},int32:{serializedBytes(){return 4},alignment(){return 4},serializeCode(e,t){return`dv.setInt32(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getInt32(${t}, isLittleEndian);`},deserializeJson(e){return Tr(e)},serializeJson(e){return e}},uint16:{serializedBytes(){return 2},alignment(){return 2},serializeCode(e,t){return`dv.setUint16(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getUint16(${t}, isLittleEndian);`},deserializeJson(e){return Tr(e)},serializeJson(e){return e}},int16:{serializedBytes(){return 2},alignment(){return 2},serializeCode(e,t){return`dv.setInt16(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getInt16(${t}, isLittleEndian);`},deserializeJson(e){return Tr(e)},serializeJson(e){return e}},uint8:{serializedBytes(){return 1},alignment(){return 1},serializeCode(e,t){return`dv.setUint8(${t}, ${e});`},deserializeCode(e,t){return`${e} = dv.getUint8(${t});`},deserializeJson(e){return Tr(e)},serializeJson(e){return e}},int8:{serializedBytes(){return 1},alignment(){return 1},serializeCode(e,t){return`dv.setInt8(${t}, ${e});`},deserializeCode(e,t){return`${e} = dv.getInt8(${t});`},deserializeJson(e){return Tr(e)},serializeJson(e){return e}}},yM=255;function vM(e,t,r){let n=0,i=r.length,s=new Array(i),o=[];for(let h=0;h<i;++h)s[h]=h;let a=h=>Hu[r[h].type].alignment(e);s.sort((h,p)=>a(p)-a(h));let c=0,l=new Array(i),u=t,f=()=>{u+=(4-u%4)%4,n+=u,o[c]=u,u=0,++c};for(let h=0;h<i;++h){let p=s[h],d=r[p],m=Hu[d.type],g=m.serializedBytes(e),y=m.alignment(e),I=(y-u%y)%y,E=u+I+g;E+(4-E%4)%4<=yM?u+=I:f(),l[p]={offset:u,group:c},u+=g}return f(),{serializedBytes:n,offsets:l,propertyGroupBytes:o}}var os=class{constructor(t,r,n){if(this.rank=t,this.firstGroupInitialOffset=r,this.propertySpecs=n,n.length===0){this.serializedBytes=r,this.serialize=this.deserialize=()=>{},this.propertyGroupBytes=[r];return}let{serializedBytes:i,offsets:s,propertyGroupBytes:o}=vM(t,r,n);this.propertyGroupBytes=o;let a="let groupOffset0 = offset;";for(let f=1;f<o.length;++f)a+=`let groupOffset${f} = groupOffset${f-1} + ${o[f-1]}*annotationCount;`;for(let f=0;f<o.length;++f)a+=`groupOffset${f} += ${o[f]}*annotationIndex;`;let c=a,l=a,u=n.length;for(let f=0;f<u;++f){let{group:h,offset:p}=s[f],d=n[f],m=Hu[d.type],g=`properties[${f}]`,y=`groupOffset${h} + ${p}`;c+=m.serializeCode(g,y,t),l+=m.deserializeCode(g,y,t)}this.serializedBytes=i,this.serialize=new Function("dv","offset","annotationIndex","annotationCount","isLittleEndian","properties",c),this.deserialize=new Function("dv","offset","annotationIndex","annotationCount","isLittleEndian","properties",l)}serializedBytes;serialize;deserialize;propertyGroupBytes};function L0(e,t){let r=[];for(let n of br){let i=as[n];r[n]=new os(e,i.serializedBytes(e),t)}return r}function Wu(e,t,r,n,i){for(let s=0;s<n;++s)e.setFloat32(t,i[s],r),t+=4;return t}function Zo(e,t,r,n,i,s){return t=Wu(e,t,r,n,i),t=Wu(e,t,r,n,s),t}function Wn(e,t,r,n,i){for(let s=0;s<n;++s)i[s]=e.getFloat32(t,r),t+=4;return t}function Ju(e,t,r,n,i,s){return t=Wn(e,t,r,n,i),t=Wn(e,t,r,n,s),t}function xM(e,t,r,n,i,s){let o=t;for(let a=0;a<s;++a)i[a]=new Float32Array(n),o=Wn(e,o,r,n,i[a]);return o}var as={1:{icon:"\uA579",description:"Line",toJSON(e){return{pointA:Array.from(e.pointA),pointB:Array.from(e.pointB)}},restoreState(e,t,r){e.pointA=re(t,"pointA",n=>We(new Float32Array(r),n,rr)),e.pointB=re(t,"pointB",n=>We(new Float32Array(r),n,rr))},serializedBytes(e){return 2*4*e},serialize(e,t,r,n,i){Zo(e,t,r,n,i.pointA,i.pointB)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n),o=new Float32Array(n);return Ju(e,t,r,n,s,o),{type:1,pointA:s,pointB:o,id:i,properties:[]}},visitGeometry(e,t){t(e.pointA,!1),t(e.pointB,!1)},defaultProperties(e){return{properties:[],values:[]}}},4:{icon:"\u2924",description:"Polyline",toJSON(e){return{points:e.points.map(t=>Array.from(t))}},restoreState(e,t,r){e.points=re(t,"points",n=>ct(n,i=>We(new Float32Array(r),i,rr)))},serializedBytes(e){return 4*e*2+4},serialize:(e,t,r,n,i,s)=>{for(let o=0;o<i.points.length-1;o++){let a=o===i.points.length-2?1:0,c=t+o*s;e.setUint32(c,a<<31|o,r);let l=i.points[o],u=i.points[o+1];Zo(e,c+4,r,n,l,u)}},deserialize:(e,t,r,n,i,s)=>{if(s===void 0)throw new Error("Can't deserialize polyline without stride");let o=new Array;if(s==0){let a=e.getUint32(t,r)&2147483647;xM(e,t+4,r,n,o,a)}else{let a=t,c=0,l=1e5;for(;c<=l;){let u=e.getUint32(a,r)>>31,f=new Float32Array(n),h=Wn(e,a+4,r,n,f);if(o.push(f),u){let p=new Float32Array(n);Wn(e,h,r,n,p),o.push(p);break}c++,a+=s}if(c===l)throw new Error("Reached max iters on polyline deserializing")}return{type:4,points:o,id:i,properties:[]}},visitGeometry(e,t){for(let r of e.points)t(r,!1)},defaultProperties(e){return{properties:[{type:"uint32",identifier:"Num vertices",default:0,description:"Number of points in the polyline"}],values:[e.points.length]}}},0:{icon:"\u26AC",description:"Point",toJSON:e=>({point:Array.from(e.point)}),restoreState:(e,t,r)=>{e.point=re(t,"point",n=>We(new Float32Array(r),n,rr))},serializedBytes:e=>e*4,serialize:(e,t,r,n,i)=>{Wu(e,t,r,n,i.point)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n);return Wn(e,t,r,n,s),{type:0,point:s,id:i,properties:[]}},visitGeometry(e,t){t(e.point,!1)},defaultProperties(e){return{properties:[],values:[]}}},2:{icon:"\u2751",description:"Bounding Box",toJSON:e=>({pointA:Array.from(e.pointA),pointB:Array.from(e.pointB)}),restoreState:(e,t,r)=>{e.pointA=re(t,"pointA",n=>We(new Float32Array(r),n,rr)),e.pointB=re(t,"pointB",n=>We(new Float32Array(r),n,rr))},serializedBytes:e=>2*4*e,serialize(e,t,r,n,i){Zo(e,t,r,n,i.pointA,i.pointB)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n),o=new Float32Array(n);return Ju(e,t,r,n,s,o),{type:2,pointA:s,pointB:o,id:i,properties:[]}},visitGeometry(e,t){t(e.pointA,!1),t(e.pointB,!1)},defaultProperties(e){return{properties:[],values:[]}}},3:{icon:"\u25CE",description:"Ellipsoid",toJSON:e=>({center:Array.from(e.center),radii:Array.from(e.radii)}),restoreState:(e,t,r)=>{e.center=re(t,"center",n=>We(new Float32Array(r),n,rr)),e.radii=re(t,"radii",n=>We(new Float32Array(r),n,ag))},serializedBytes:e=>2*4*e,serialize(e,t,r,n,i){Zo(e,t,r,n,i.center,i.radii)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n),o=new Float32Array(n);return Ju(e,t,r,n,s,o),{type:3,center:s,radii:o,id:i,properties:[]}},visitGeometry(e,t){t(e.center,!1),t(e.radii,!0)},defaultProperties(e){return{properties:[],values:[]}}}};function SM(e,t){let r=0,n=[],i=[];for(let f of br){let p=t[f].serializedBytes;n[f]=r;let d=e[f],m=d.length;if(f===4){i[f]=0;for(let g of d){let y=g.points.length-1;r+=p*y,i[f]+=y}}else r+=p*m}let s=[],o=[],a=[],c=new ArrayBuffer(r),l=new DataView(c),u=Sr===oe.LITTLE;for(let f of br){let h=t[f],{rank:p}=h,d=h.serialize,m=e[f];a[f]=Array.from({length:m.length},(b,C)=>C),s[f]=m.map(b=>b.id),o[f]=new Map(m.map((b,C)=>[b.id,C]));let y=as[f].serialize,I=n[f],_=h.propertyGroupBytes[0],E=0;for(let b=0,C=m.length;b<C;++b){let v=m[b];if(f===4){let w=v;y(l,I+E*_,u,p,w,_),a[f][b]=E;for(let x=0;x<w.points.length-1;x++)d(l,I,E+x,i[f],u,w.properties);E+=w.points.length-1}else y(l,I+b*_,u,p,v,_),d(l,I,b,C,u,v.properties)}f!==4&&(i[f]=m.length)}return{data:new Uint8Array(c),typeToInstanceCounts:a,typeToIds:s,typeToOffset:n,typeToIdMaps:o,typeToSize:i}}var Qo=class{constructor(t){this.propertySerializers=t}annotations=[[],[],[],[],[]];add(t){this.annotations[t.type].push(t)}serialize(){return SM(this.annotations,this.propertySerializers)}};function F0(e,t){if(!e.accessToken)return t;let r=new Headers(t.headers);return r.set("Authorization",`${e.tokenType} ${e.accessToken}`),{...t,headers:r}}function B0(e,t){let{status:r}=e;if(r===401||r===403&&!t.accessToken)return"refresh";throw e instanceof Error&&t.email!==void 0&&(e.message+=`  (Using credentials for ${JSON.stringify(t.email)})`),e}function z0(e,t,r){return e===void 0?Re(t,r):ln(e,t,r,F0,B0)}function ea(e){return e===void 0?Re:Hy(e,F0,B0)}function Dt(e,t,r,n={}){return z0(t,`${e.serverUrl}${r}`,n)}var zr=(e=>(e[e.RAW=0]="RAW",e[e.JPEG=1]="JPEG",e[e.COMPRESSED_SEGMENTATION=2]="COMPRESSED_SEGMENTATION",e))(zr||{});var ta=class{instance;volumeId;scaleIndex;encoding;jpegQuality;changeSpec;static RPC_ID="brainmaps/VolumeChunkSource"},ra=class{instance;volumeId;info;changeSpec;static RPC_ID="brainmaps/MultiscaleMeshSource"},na=class{instance;volumeId;meshName;changeSpec;static RPC_ID="brainmaps/MeshSource"},ia=class{instance;volumeId;meshName;changeSpec;static RPC_ID="brainmaps/SkeletonSource"},sa=class{instance;volumeId;changestack;upperVoxelBound;static RPC_ID="brainmaps/Annotation"},oa=class{instance;volumeId;changestack;static RPC_ID="brainmaps/AnnotationSpatialIndex"};var $0="skeleton/SkeletonLayer";var wM=Object.defineProperty,EM=Object.getOwnPropertyDescriptor,bM=(e,t,r,n)=>{for(var i=n>1?void 0:n?EM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&wM(t,r,i),i};var IM=60,Xu=class extends fe{objectId=0n;vertexPositions=null;vertexAttributes=null;indices=null;initializeSkeletonChunk(t,r){super.initialize(t),this.objectId=r}freeSystemMemory(){this.vertexPositions=this.indices=null}getVertexAttributeBytes(){let t=this.vertexPositions.byteLength,{vertexAttributes:r}=this;return r?.forEach(n=>{t+=n.byteLength}),t}serialize(t,r){super.serialize(t,r);let n=this.vertexPositions,i=this.indices;t.numVertices=n.length/3,t.indices=i,r.push(i.buffer);let{vertexAttributes:s}=this;if(s!=null&&s.length>0){let o=new Uint8Array(this.getVertexAttributeBytes());o.set(new Uint8Array(n.buffer,n.byteOffset,n.byteLength));let a=t.vertexAttributeOffsets=new Uint32Array(s.length+1);a[0]=0;let c=n.byteLength;s.forEach((l,u)=>{a[u+1]=c,o.set(new Uint8Array(l.buffer,l.byteOffset,l.byteLength),c),c+=l.byteLength}),r.push(o.buffer),t.vertexAttributes=o}else t.vertexAttributes=new Uint8Array(n.buffer,n.byteOffset,n.byteLength),t.vertexAttributeOffsets=Uint32Array.of(0),n.buffer!==r[0]&&r.push(n.buffer);this.vertexPositions=this.indices=this.vertexAttributes=null}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=this.indices.byteLength+this.getVertexAttributeBytes(),super.downloadSucceeded()}},$r=class extends Ne{getChunk(t){let r=on(t),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(Xu),n.initializeSkeletonChunk(r,t),this.addChunk(n)),n}},V0=class extends an(Et(Xe(Dr))){source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;this.chunkManager.registerLayer(this);let t=je(e),r=Ge(e),{source:n,chunkManager:i}=this;xr(this,s=>{let o=n.getChunk(s);++this.numVisibleChunksNeeded,o.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable,i.requestChunk(o,t,r+IM)})}};V0=bM([j($0)],V0);function aa(e,t,r,n,i,s,o){let a=Ou(2,t,r,n,i,s,o);e.vertexPositions=a.vertexPositions,e.indices=a.indices}async function Xn(e,t,r){e.data=new Uint32Array(r)}async function ft(e,t,r,n=Sr,i=0,s=r.byteLength){let{spec:o}=e.source,{dataType:a}=o,c=Qi(e.chunkDataSize),l=lt[a],u=c*l;if(u!==s)throw new Error(`Raw-format chunk is ${s} bytes, but ${c} * ${l} = ${u} bytes are expected.`);let f=Co(a,r,i,s);qn(f,n,l),await ut(e,t,f)}var _M=Object.defineProperty,CM=Object.getOwnPropertyDescriptor,Zn=(e,t,r,n)=>{for(var i=n>1?void 0:n?CM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&_M(t,r,i),i};var AM=new Map([[zr.RAW,ft],[zr.JPEG,Br],[zr.COMPRESSED_SEGMENTATION,Xn]]);function Z0(e,t){e&&(t.change_spec={change_stack_id:e.changeStackId},e.timeStamp&&(t.change_spec.time_stamp=e.timeStamp),e.skipEquivalences&&(t.change_spec.skip_equivalences=e.skipEquivalences))}function Qn(e,t){return ye(Nn()(e),t)}var j0=class extends Qn(Ae,ta){chunkDecoder=AM.get(this.parameters.encoding);applyEncodingParams(e){let{encoding:t}=this.parameters;switch(t){case zr.RAW:e.subvolume_format="RAW";break;case zr.JPEG:e.subvolume_format="SINGLE_IMAGE",e.image_format_options={image_format:"JPEG",jpeg_quality:this.parameters.jpegQuality};return;case zr.COMPRESSED_SEGMENTATION:e.subvolume_format="RAW",e.image_format_options={compressed_segmentation_block_size:tr(this.spec.compressedSegmentationBlockSize)};break;default:throw new Error(`Invalid encoding: ${t}`)}}async download(e,t){let{parameters:r}=this,n=this.computeChunkBounds(e),i=e.chunkDataSize,s=`/v1/volumes/${r.volumeId}/subvolume:binary`,o={geometry:{corner:tr(n),size:tr(i),scale:r.scaleIndex}};this.applyEncodingParams(o),Z0(r.changeSpec,o);let a=await Dt(r.instance,this.credentialsProvider,s,{method:"POST",body:JSON.stringify(o),signal:t});await this.chunkDecoder(e,t,await a.arrayBuffer())}};j0=Zn([j()],j0);function MM(e,t,r,n){let i=yr(BigInt("0x"+e));return t0(i,t,r,n)}function kM(e,t){Ee(t);let r=e.source,n=re(t,"fragmentKey",Yt),i=re(t,"supervoxelId",Yt),s=n.length;if(s!==i.length)throw new Error("Expected fragmentKey and supervoxelId arrays to have the same length.");let o=new Map;n.forEach((E,b)=>{let C=o.get(E);C===void 0&&(C=[],o.set(E,C)),C.push(i[b])});let{chunkShape:a}=r.parameters.info,c=r.parameters.info.lods[0].gridShape,l=Math.ceil(Math.log2(c[0])),u=Math.ceil(Math.log2(c[1])),f=Math.ceil(Math.log2(c[2])),h=Array.from(o.entries()).map(([E,b])=>({fragmentId:E,corner:MM(E,l,u,f),supervoxelIds:b}));h.sort((E,b)=>Gn(E.corner[0],E.corner[1],E.corner[2],b.corner[0],b.corner[1],b.corner[2])?-1:1);let p,d,m=0,g;if(s===0)p=d=qi,g=Uint32Array.of(0,0,0,0,2147483648);else{let E=B.clone(tg),b=B.clone(qi);for(h.forEach(C=>{let{corner:v}=C;for(let w=0;w<3;++w)E[w]=Math.min(E[w],v[w]),b[w]=Math.max(b[w],v[w])}),m=1;b[0]>>>m-1!==E[0]>>>m-1||b[1]>>>m-1!==E[1]>>>m-1||b[2]>>>m-1!==E[2]>>>m-1;)++m;p=B.multiply(E,E,a),d=B.add(b,B.multiply(b,b,a),a)}let{lods:y}=r.parameters.info,I=new Float32Array(Math.max(y.length,m));for(let E=0;E<y.length;++E)I[E]=y[E].scale;if(s!==0){let E=new Uint32Array(h.length*I.length*5);h.forEach((v,w)=>{E.set(v.corner,w*5),E[w*5]=v.corner[0]});let b=0,C=h.length;for(let v=1;v<I.length;++v){let w=Ho(E,b,C);b=C,C=w}g=E.slice(0,C*5)}let _={chunkShape:a,chunkGridSpatialOrigin:qi,clipLowerBound:p,clipUpperBound:d,octree:g,lodScales:I,vertexOffsets:new Float32Array(I.length*3)};e.manifest=_,e.fragmentSupervoxelIds=h}var Qu=255;function Q0(e,t){let r=e.byteLength,n=0,i=new DataView(e),s=32;for(;n<r;){if(n+s>r)throw new Error("Invalid batch mesh fragment response.");let c=i.getBigUint64(n,!0).toString()+"\0";n+=8;let l=i.getUint32(n,!0),u=i.getUint32(n+4,!0);if(n+=8,u!==0)throw new Error("Invalid batch mesh fragment response.");if(n+l+8+8>r)throw new Error("Invalid batch mesh fragment response.");let f=new TextDecoder().decode(new Uint8Array(e,n,l)),h=c+f;n+=l;let p=i.getUint32(n,!0),d=i.getUint32(n+4,!0);n+=8;let m=i.getUint32(n,!0),g=i.getUint32(n+4,!0);if(n+=8,d!==0||g!==0)throw new Error("Invalid batch mesh fragment response.");let y=n+m*12+p*12;if(y>r)throw new Error("Invalid batch mesh fragment response.");t({fullKey:h,buffer:e,verticesOffset:n,numVertices:p,indicesOffset:n+12*p,numIndices:m*3}),n=y}}function ev(e){let t=0,r=0;for(let a of e)t+=a.numVertices,r+=a.numIndices;let n=new Float32Array(t*3),i=new Uint32Array(r),s=0,o=0;for(let a of e){n.set(new Float32Array(a.buffer,a.verticesOffset,a.numVertices*3),s*3);let{numIndices:c}=a,l=new Uint32Array(a.buffer,a.indicesOffset,c);It(l,oe.LITTLE);for(let u=0;u<c;++u)i[o++]=l[u]+s;s+=a.numVertices}return It(n,oe.LITTLE),{vertexPositions:n,indices:i}}async function tv(e,t,r,n){let i="/v1/objects/meshes:batch",s=[],o,a=0,c=new Map;for(let[u,f]of r){c.set(u,f),r.delete(u);let h=u.indexOf("\0"),p=u.substring(0,h),d=u.substring(h+1);if(p!==o&&s.push({object_id:p,fragment_keys:[]}),s[s.length-1].fragment_keys.push(d),++a===Qu)break}let l={volume_id:t.volumeId,mesh_name:t.meshName,batches:s};try{return await(await Dt(t.instance,e,i,{method:"POST",body:JSON.stringify(l),signal:n})).arrayBuffer()}finally{for(let[u,f]of c)r.set(u,f)}}var G0=class extends Qn(Kn,ra){listFragmentsParams=(()=>{let{parameters:e}=this,{changeSpec:t}=e;return t!==void 0?`&header.changeStackId=${t.changeStackId}`:""})();download(e,t){let{parameters:r}=this,n=`/v1/objects/${r.volumeId}/meshes/${r.info.lods[0].info.name}:listfragments?object_id=${e.objectId}&return_supervoxel_ids=true`+this.listFragmentsParams;return Dt(r.instance,this.credentialsProvider,n,{signal:t}).then(i=>i.json()).then(i=>kM(e,i))}async downloadFragment(e,t){let{parameters:r}=this,n=e.manifestChunk,{fragmentSupervoxelIds:i}=n,s=n.manifest,{lod:o}=e,{octree:a}=s,c=i.length,l=e.chunkIndex,u=l;for(;u>=c;)u=a[u*5+3];let f=l+1;for(;f>c;)f=a[f*5-1]&2147483647;let{relativeBlockShape:h,gridShape:p}=r.info.lods[o],d=Math.ceil(Math.log2(p[0])),m=Math.ceil(Math.log2(p[1])),g=Math.ceil(Math.log2(p[2])),y=new Map;for(let M=u;M<f;++M){let P=Math.floor(a[M*5]/h[0]),F=Math.floor(a[M*5+1]/h[1]),S=Math.floor(a[M*5+2]/h[2]),O=$o(d,m,g,P,F,S).toString(16).padStart(16,"0"),R=i[M];for(let N of R.supervoxelIds)y.set(N+"\0"+O,M)}let I=Math.max(0,o-1),_=[],E=Array.from(y);E.sort((M,P)=>St(M[0],P[0])),y=new Map(E);let b=r.info.lods[o].info.name,C=!0;await new Promise((M,P)=>{let F=0,S=!1,O=()=>{if(!S){for(;y.size!==0&&(++F,tv(this.credentialsProvider,{instance:r.instance,volumeId:r.volumeId,meshName:b},y,t).then(R=>{--F,Q0(R,N=>{let U=y.get(N.fullKey);if(!y.delete(N.fullKey))throw new Error(`Received unexpected fragment key: ${JSON.stringify(N.fullKey)}.`);N.chunkIndex=U,_.push(N)}),O()}).catch(R=>{S=!0,P(R)}),!!C););if(e.downloadSlots=Math.max(1,F),F===0){M(void 0);return}}};O()}),_.sort((M,P)=>M.chunkIndex-P.chunkIndex);let v=0,w=1<<3*(o-I),x=new Uint32Array(w+1),k=0;for(let M of _){let P=M.chunkIndex,F=Cu(a[P*5]>>>I,a[P*5+1]>>>I,a[P*5+2]>>>I)&w-1;x.fill(v,k+1,F+1),k=F,v+=M.numIndices}x.fill(v,k+1,w+1),Jo(e,{...ev(_),subChunkOffsets:x},un.float32)}};G0=Zn([j()],G0);function TM(e){let t=[],r=0,n=e.length;for(;r<n;)t.push(JSON.stringify(e.slice(r,r+Qu))),r+=Qu;return t}function NM(e,t){Ee(t);let r=re(t,"fragmentKey",Yt),n=re(t,"supervoxelId",Yt);if(r.length!==n.length)throw new Error("Expected fragmentKey and supervoxelId arrays to have the same length.");let s=n.map((o,a)=>o+"\0"+r[a]);e.fragmentIds=TM(s)}var q0=class extends Qn(Nt,na){listFragmentsParams=(()=>{let{parameters:e}=this,{changeSpec:t}=e;return t!==void 0?`&header.changeStackId=${t.changeStackId}`:""})();download(e,t){let{parameters:r}=this,n=`/v1/objects/${r.volumeId}/meshes/${r.meshName}:listfragments?object_id=${e.objectId}&return_supervoxel_ids=true`+this.listFragmentsParams;return Dt(r.instance,this.credentialsProvider,n,{signal:t}).then(i=>i.json()).then(i=>NM(e,i))}async downloadFragment(e,t){let{parameters:r}=this,n=new Map;for(let o of JSON.parse(e.fragmentId))n.set(o,null);let i=[],{credentialsProvider:s}=this;for(;n.size!==0;){let o=await tv(s,r,n,t);Q0(o,a=>{if(!n.delete(a.fullKey))throw new Error(`Received unexpected fragment key: ${JSON.stringify(a.fullKey)}.`);i.push(a)})}or(e,ev(i))}};q0=Zn([j()],q0);function DM(e,t){let r=new DataView(t),n=r.getUint32(0,!0);if(r.getUint32(4,!0)!==0)throw new Error("The number of vertices should not exceed 2^32-1.");let s=r.getUint32(8,!0);if(r.getUint32(12,!0)!==0)throw new Error("The number of edges should not exceed 2^32-1.");aa(e,t,oe.LITTLE,16,n,void 0,s)}var K0=class extends Qn($r,ia){download(e,t){let{parameters:r}=this,n={object_id:`${e.objectId}`},i=`/v1/objects/${r.volumeId}/meshes/${r.meshName}/skeleton:binary`;return Z0(r.changeSpec,n),Dt(r.instance,this.credentialsProvider,i,{method:"POST",body:JSON.stringify(n),signal:t}).then(s=>s.arrayBuffer()).then(s=>DM(e,s))}};K0=Zn([j()],K0);var tf=["LOCATION","LINE","VOLUME"];function Y0(e){let t=/(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)/,r=e.match(t);if(r===null)throw new Error(`Error parsing number triplet: ${JSON.stringify(e)}.`);return B.fromValues(parseFloat(r[1]),parseFloat(r[2]),parseFloat(r[3]))}function ef(e){return e.volumeId+":"+e.changestack+":"}function rv(e,t){if(!t.startsWith(e))throw new Error(`Received annotation id ${JSON.stringify(t)} does not have expected prefix of ${JSON.stringify(e)}.`);return t.substring(e.length)}function PM(e){if(e!=null)return[BigUint64Array.from(ct(e,yr))]}function nv(e,t,r){let n=re(e,"corner",u=>Y0(et(u))),i=re(e,"size",u=>Y0(et(u))),s=re(e,"payload",Jl),o=re(e,"type",et),a=re(e,"id",et),c=rv(t,a),l=re(e,"objectLabels",PM);if(r!==void 0&&c!==r)throw new Error(`Received annotation has unexpected id ${JSON.stringify(a)}.`);switch(o){case"LOCATION":{if(B.equals(i,qi))return{type:Oe.POINT,id:c,point:n,description:s,relatedSegments:l,properties:[]};let u=B.scale(B.create(),i,.5),f=B.add(B.create(),n,u);return{type:Oe.ELLIPSOID,id:c,center:f,radii:u,description:s,relatedSegments:l,properties:[]}}case"LINE":return{type:Oe.LINE,id:c,pointA:n,pointB:B.add(B.create(),n,i),description:s,relatedSegments:l,properties:[]};case"VOLUME":return{type:Oe.AXIS_ALIGNED_BOUNDING_BOX,id:c,pointA:n,pointB:B.add(B.create(),n,i),description:s,relatedSegments:l,properties:[]};default:throw new Error(`Unknown spatial annotation type: ${JSON.stringify(o)}.`)}}function RM(e,t,r){Ee(e);let n=re(e,"annotations",i=>We([void 0],i,Ee))[0];return nv(n,t,r)}var OM=L0(3,[]);function iv(e,t){let r=new Qo(OM),n=e.source.parent,i=ef(n.parameters);t.forEach((s,o)=>{try{Ee(s);let a=re(s,"annotations",c=>c===void 0?[]:c);if(!Array.isArray(a))throw new Error(`Expected array, but received ${JSON.stringify(typeof a)}.`);for(let c of a)try{r.add(nv(c,i))}catch(l){throw new Error(`Error parsing annotation: ${l.message}`)}}catch(a){throw new Error(`Error parsing ${tf[o]} annotations: ${a.message}`)}}),e.data=Object.assign(new Vn,r.serialize())}function J0(e){let t=e.indexOf(".");return e.substring(0,t)}function fn(e){return`${Math.round(e[0])},${Math.round(e[1])},${Math.round(e[2])}`}function Zu(e,t){return`${e.volumeId}:${e.changestack}:${t}`}function H0(e){let t=e.description||"",r=e.relatedSegments===void 0?void 0:Array.from(e.relatedSegments[0],n=>n.toString());switch(e.type){case Oe.LINE:{let{pointA:n,pointB:i}=e,s=B.subtract(B.create(),i,n);return{type:"LINE",corner:fn(n),size:fn(s),object_labels:r,payload:t}}case Oe.AXIS_ALIGNED_BOUNDING_BOX:{let{pointA:n,pointB:i}=e,s=cu(B.create(),n,i),o=$g(B.create(),n,i),a=B.subtract(o,o,s);return{type:"VOLUME",corner:fn(s),size:fn(a),object_labels:r,payload:t}}case Oe.POINT:return{type:"LOCATION",corner:fn(e.point),size:"0,0,0",object_labels:r,payload:t};case Oe.ELLIPSOID:{let n=B.subtract(B.create(),e.center,e.radii),i=B.scale(B.create(),e.radii,2);return{type:"LOCATION",corner:fn(n),size:fn(i),object_labels:r,payload:t}}}}var W0=class extends Qn(cn,oa){async download(e,t){let{parameters:r}=this;return Promise.all(tf.map(n=>Dt(r.instance,this.credentialsProvider,`/v1/changes/${r.volumeId}/${r.changestack}/spatials:get`,{signal:t,method:"POST",body:JSON.stringify({type:n,ignore_payload:!0})}).then(i=>i.json()))).then(n=>{iv(e,n)})}};W0=Zn([j()],W0);var X0=class extends Qn(jn,sa){downloadSegmentFilteredGeometry(e,t,r){let{parameters:n}=this;return Promise.all(tf.map(i=>Dt(n.instance,this.credentialsProvider,`/v1/changes/${n.volumeId}/${n.changestack}/spatials:get`,{signal:r,method:"POST",body:JSON.stringify({type:i,object_labels:[e.objectId.toString()],ignore_payload:!0})}).then(s=>s.json()))).then(i=>{iv(e,i)})}downloadMetadata(e,t){let{parameters:r}=this,n=e.key;return Dt(r.instance,this.credentialsProvider,`/v1/changes/${r.volumeId}/${r.changestack}/spatials:get`,{signal:t,method:"POST",body:JSON.stringify({type:J0(n),id:Zu(r,n)})}).then(i=>i.json()).then(i=>{e.annotation=RM(i,ef(r),n)},()=>{e.annotation=null})}add(e){let{parameters:t}=this,r=H0(e);return Dt(t.instance,this.credentialsProvider,`/v1/changes/${t.volumeId}/${t.changestack}/spatials:push`,{method:"POST",body:JSON.stringify({annotations:[r]})}).then(n=>n.json()).then(n=>{Ee(n);let i=re(n,"ids",Yt);if(i.length!==1)throw new Error(`Expected list of 1 id, but received ${JSON.stringify(i)}.`);let s=ef(this.parameters);return rv(s,i[0])})}update(e,t){let{parameters:r}=this,n=H0(t);return n.id=Zu(r,e),Dt(r.instance,this.credentialsProvider,`/v1/changes/${r.volumeId}/${r.changestack}/spatials:push`,{method:"POST",body:JSON.stringify({annotations:[n]})}).then(i=>i.json())}delete(e){let{parameters:t}=this;return Dt(t.instance,this.credentialsProvider,`/v1/changes/${t.volumeId}/${t.changestack}/spatials:delete`,{method:"POST",body:JSON.stringify({type:J0(e),ids:[Zu(t,e)]})}).then(r=>r.json())}};X0=Zn([j()],X0);var hn=De("decodePng");var cs=(e=>(e[e.JPG=0]="JPG",e[e.JPEG=1]="JPEG",e[e.PNG=2]="PNG",e))(cs||{}),ca=class{url;encoding;format;tilesize;overlap;static RPC_ID="deepzoom/ImageTileSource"};var UM=Object.defineProperty,LM=Object.getOwnPropertyDescriptor,FM=(e,t,r,n)=>{for(var i=n>1?void 0:n?LM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&UM(t,r,i),i};var sv=class extends ye(Te(Ae),ca){tileKvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);gridShape=(()=>{let e=new Uint32Array(2),{upperVoxelBound:t,chunkDataSize:r}=this.spec;for(let n=0;n<2;++n)e[n]=Math.ceil(t[n]/r[n]);return e})();async download(e,t){let{parameters:r}=this,{tilesize:n,overlap:i,encoding:s}=r,[o,a]=e.chunkGridPosition,c=o===0?0:i,l=a===0?0:i,u=`${this.tileKvStore.path}/${o}_${a}.${r.format}`,f=await this.tileKvStore.store.read(u,{signal:t});if(f===void 0)return;let h=new Uint8Array(await f.response.arrayBuffer()),p=0,d=0,m;switch(s){case cs.PNG:{let g=await ue(hn,t,[h.buffer],h,void 0,void 0,void 0,3,1,!1);({width:p,height:d}=g),m=Qm(g.uint8Array,p*d,3);break}case cs.JPG:case cs.JPEG:{({uint8Array:m,width:p,height:d}=await ue(Hn,t,[h.buffer],h,void 0,void 0,void 0,3,!1));break}}if(m!==void 0){let g=n*n,y=p*d,I=e.data=new Uint8Array(g*3);for(let _=0;_<3;_++)for(let E=0;E<d;E++)for(let b=0;b<p;b++)I[b+E*n+_*g]=m[b+c+(E+l)*p+_*y]}}};sv=FM([j()],sv);var la=class{constructor(t,r){this.baseUrl=t,this.nodeKey=r}getNodeApiUrl(t=""){return`${this.baseUrl}/api/node/${this.nodeKey}${t}`}getRepoInfoUrl(){return`${this.baseUrl}/api/repos/info`}getKeyValueUrl(t,r){return`${this.getNodeApiUrl()}/${t}/key/${r}`}getKeyValueRangeUrl(t,r,n){return`${this.getNodeApiUrl()}/${t}/keyrange/${r}/${n}`}getKeyValuesUrl(t){return`${this.getNodeApiUrl()}/${t}/keyvalues?jsontar=false`}};function ua(e,t){return e.includes("?")?e+="&":e+="?",e+="app=Neuroglancer",t&&(e+=`&u=${t}`),e}function fa(e,t,r){return ln(e,t,r,(n,i)=>{let s={...i};return n.token&&(s.headers={...s.headers,Authorization:`Bearer ${n}`}),s},n=>{let{status:i}=n;if(i===403||i===401)return"refresh";throw n})}var Vr=(e=>(e[e.JPEG=0]="JPEG",e[e.RAW=1]="RAW",e[e.COMPRESSED_SEGMENTATION=2]="COMPRESSED_SEGMENTATION",e[e.COMPRESSED_SEGMENTATIONARRAY=3]="COMPRESSED_SEGMENTATIONARRAY",e))(Vr||{}),ls=class{baseUrl;nodeKey;dataInstanceKey;authServer;user},ha=class extends ls{dataScale;encoding;static RPC_ID="dvid/VolumeChunkSource"},pa=class extends ls{static RPC_ID="dvid/SkeletonSource"},da=class extends ls{static RPC_ID="dvid/MeshSource"};function ov(e,t){let r=BM(t);if(r.length<1)throw new Error("ERROR parsing swc data");let n=new Uint32Array(r.length),i=0,s=0;r.forEach((u,f)=>{u&&(n[f]=i++,u.parent>=0&&++s)});let o=new Float32Array(3*i),a=new Uint32Array(2*s),c=0,l=0;r.forEach(u=>{u&&(o[3*c]=u.x,o[3*c+1]=u.y,o[3*c+2]=u.z,u.parent>=0&&(a[2*l]=c,a[2*l+1]=n[u.parent],++l),++c)}),e.indices=a,e.vertexPositions=o}function BM(e){let t=e.split(`
`),r=[],n="-?\\d*(?:\\.\\d+)?",i=new RegExp("^[ \\t]*("+["\\d+","\\d+",n,n,n,n,"-1|\\d+"].join(")[ \\t]+(")+")[ \\t]*$");return t.forEach(s=>{let o=s.match(i);if(o){let a=r[parseInt(o[1],10)]=new rf;a.type=parseInt(o[2],10),a.x=parseFloat(o[3]),a.y=parseFloat(o[4]),a.z=parseFloat(o[5]),a.radius=parseFloat(o[6]),a.parent=parseInt(o[7],10)}}),r}var rf=class{type;x;y;z;radius;parent};var zM=Object.defineProperty,$M=Object.getOwnPropertyDescriptor,nf=(e,t,r,n)=>{for(var i=n>1?void 0:n?$M(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&zM(t,r,i),i};function sf(e,t){return ye(Nn()(e),t)}var av=class extends sf($r,pa){download(e,t){let{parameters:r}=this,n=`${e.objectId}`,i=`${r.baseUrl}/api/node/${r.nodeKey}/${r.dataInstanceKey}/key/`+n+"_swc";return fa(this.credentialsProvider,ua(i,r.user),{signal:t}).then(s=>s.arrayBuffer()).then(s=>{let o=new TextDecoder("utf-8");ov(e,o.decode(s))})}};av=nf([j()],av);function VM(e,t){let n=new DataView(t).getUint32(0,!0);or(e,Yn(t,oe.LITTLE,4,n))}var cv=class extends sf(Nt,da){download(e){return e.fragmentIds=[`${e.objectId}`],Promise.resolve(void 0)}downloadFragment(e,t){let{parameters:r}=this,i=new la(r.baseUrl,r.nodeKey).getKeyValueUrl(r.dataInstanceKey,`${e.fragmentId}.ngmesh`);return fa(this.credentialsProvider,ua(i,r.user),{signal:t}).then(s=>s.arrayBuffer()).then(s=>VM(e,s))}};cv=nf([j()],cv);var lv=class extends sf(Ae,ha){async download(e,t){let r=this.parameters,n;{let o=this.computeChunkBounds(e),a=e.chunkDataSize;n=this.getPath(o,a)}let i=this.getDecoder(r),s=await fa(this.credentialsProvider,ua(`${r.baseUrl}${n}`,r.user),{signal:t}).then(o=>o.arrayBuffer());await i(e,t,r.encoding===Vr.JPEG?s.slice(16):s)}getPath(e,t){let r=this.parameters;return r.encoding===Vr.JPEG?`/api/node/${r.nodeKey}/${r.dataInstanceKey}/subvolblocks/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}`:r.encoding===Vr.RAW?`/api/node/${r.nodeKey}/${r.dataInstanceKey}/raw/0_1_2/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}/jpeg`:r.encoding===Vr.COMPRESSED_SEGMENTATIONARRAY?`/api/node/${r.nodeKey}/${r.dataInstanceKey}/raw/0_1_2/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}?compression=googlegzip&scale=${r.dataScale}`:`/api/node/${r.nodeKey}/${r.dataInstanceKey}/raw/0_1_2/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}?compression=googlegzip`}getDecoder(e){return e.encoding===Vr.JPEG||e.encoding===Vr.RAW?Br:Xn}};lv=nf([j()],lv);function ma(e,t){if(t===void 0)return{outer:e,inner:{offset:0,length:e.length}};if("suffixLength"in t){let r=Math.min(e.length,t.suffixLength);return{outer:{offset:e.offset+(e.length-r),length:r},inner:{offset:e.length-r,length:r}}}if(t.offset+t.length>e.length)throw new Error(`Requested byte range ${JSON.stringify(t)} not valid for value of length ${e.length}`);return{outer:{offset:e.offset+t.offset,length:t.length},inner:t}}function us(e,t){let{outer:{offset:r,length:n}}=ma({offset:0,length:e.length},t);return{offset:r,length:n,totalSize:e.length,response:new Response(e.subarray(r,r+n))}}var tt=class{constructor(t,r){this.base=t,this.byteRange=r}async stat(t){return{totalSize:this.byteRange.length}}async read(t){let{byteRange:r}=this,{outer:n,inner:i}=ma(r,t.byteRange);return n.length===0?{response:new Response(new Uint8Array(0)),totalSize:r.length,...i}:{response:(await Pr(this.base,{signal:t.signal,byteRange:n,strictByteRange:!0,throwIfMissing:!0})).response,totalSize:r.length,...i}}getUrl(){let{offset:t,length:r}=this.byteRange;return`${this.base.getUrl()}|range:${t}-${t+r}`}};function of(e){if(e!==void 0)return`bytes=${e.offset}-${e.offset+e.length-1}`}var jM=navigator.userAgent.indexOf("Chrome")!==-1?"no-store":"default";function af(e,t){return new URL(e).pathname+"/"===new URL(t.url).pathname}function uv(e){let t=e.match(/bytes ([0-9]+)-([0-9]+)\/([0-9]+|\*)/);if(t===null)throw new Error(`Invalid content-range header: ${JSON.stringify(e)}`);let r=parseInt(t[1],10),n=parseInt(t[2],10),i;t[3]!=="*"&&(i=parseInt(t[3],10));let s=n-r+1;return{offset:r,length:s,totalSize:i}}async function ei(e,t,r,n,i=Re){let s;try{let{byteRange:o}=n,a;if(o!==void 0)if("suffixLength"in o){let p=await pn(e,t,r,n,i);if(p===void 0)return;let{totalSize:d}=p;if(d===void 0)throw new Error(`Failed to determine total size of ${e.getUrl(t)} in order to fetch suffix ${JSON.stringify(o)}`);if(s=ma({offset:0,length:d},o).outer,s.length===0)return{...s,totalSize:d,response:new Response(new Uint8Array(0))};a=of(s)}else s=o,s.length===0?a=of({offset:Math.max(s.offset-1,0),length:1}):a=of(s);let c={signal:n.signal,progressListener:n.progressListener};a!==void 0&&(c.headers={range:a},c.cache=jM);let l=await i(r,c);if(af(r,l))return;let u,f,h;if(l.status===206){let p=l.headers.get("content-range");if(p===null)if(s!==void 0)u=s.offset;else throw new Error("Unexpected HTTP 206 response when no byte range specified.");p!==null&&({offset:u,length:f,totalSize:h}=uv(p))}else f=h=ga(l.headers);return u===void 0&&(u=0),f===void 0&&(f=ga(l.headers)),s?.length===0&&(l=new Response(new Uint8Array(0)),u=s.offset,f=0),{response:l,offset:u,length:f,totalSize:h}}catch(o){return o instanceof bt&&o.status===416&&s?.length===0&&s.offset===0?{response:new Response(new Uint8Array(0)),offset:0,length:0,totalSize:0}:cf(e,t,n,o)}}function ga(e){let t=e.get("content-length");if(e.get("content-encoding")===null&&t!==null){let n=Number(t);if(!Number.isFinite(n)||n<0)throw new Error("Invalid content-length: {contentLength}");return n}}function cf(e,t,r,n){if(Jy(n)){if(r.throwIfMissing===!0)throw new Wi(new ke(e,t),{cause:n});return}throw n}async function pn(e,t,r,n,i=Re){try{let s=await i(r,{method:"HEAD",signal:n.signal,progressListener:n.progressListener});return af(r,s)?void 0:{totalSize:ga(s.headers)}}catch(s){if(!(s instanceof bt&&(s.status===405||s.status===501)))return cf(e,t,n,s)}try{let s=await i(r,{signal:n.signal,progressListener:n.progressListener,headers:{range:"bytes=0-0"}});if(af(r,s))return;let o;if(s.status===200)o=ga(s.headers);else{let a=s.headers.get("content-range");a!==null&&({totalSize:o}=uv(a))}return{totalSize:o}}catch(s){return s instanceof bt&&s.status===416?{totalSize:0}:cf(e,t,n,s)}}var ti=class{constructor(t,r,n=r,i=Re){this.sharedKvStoreContext=t,this.baseUrl=r,this.baseUrlForDisplay=n,this.fetchOkImpl=i}stat(t,r){return pn(this,t,Ht(this.baseUrl,t),r,this.fetchOkImpl)}read(t,r){return ei(this,t,Ht(this.baseUrl,t),r,this.fetchOkImpl)}getUrl(t){return Ht(this.baseUrlForDisplay,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function GM(e,t,r){return{scheme:e,description:`${e} (unauthenticated)`,getKvStore(n){try{let{baseUrl:i,path:s}=Dn(n.url);return{store:new r(t,i),path:s}}catch(i){throw new Error(`Invalid URL ${JSON.stringify(n.url)}`,{cause:i})}}}}function fv(e,t){for(let r of["http","https"])e.registerBaseKvStoreProvider(n=>GM(r,n,t))}var hv="GrapheneMeshSource:NewSegment";var ya=class{url;static RPC_ID="graphene/ChunkedGraphSource"},va=class{manifestUrl;fragmentUrl;lod;sharding;nBitsForLayerId;static RPC_ID="graphene/MeshSource"};function lf(e,t){return e>>BigInt(64-t)==1n}function pv(e){if(e.charAt(0)==="~"){let r=e.substring(1).split(/:(.+)/);return{key:r[0],fragmentId:r[1]}}return{key:e,fragmentId:e}}var dv="ChunkedGraphLayer",mv="ChunkedGraphLayer:updateSources",gv=5;async function yv(e){if(e.response){let t;return e.response.headers.get("content-type")==="application/json"?t=(await e.response.json()).message:t=await e.response.text(),t}}function uf(e,t){let{store:r,path:n}=e.getKvStore(t);if(!(r instanceof ti))throw new Error(`Non-HTTP URL ${JSON.stringify(t)} not supported`);let{fetchOkImpl:i,baseUrl:s}=r;if(s.includes("?"))throw new Error(`Invalid URL ${s}: query parameters not supported`);return{fetchOkImpl:i,baseUrl:Ht(s,n)}}var jr=(e=>(e[e.RAW=0]="RAW",e[e.JPEG=1]="JPEG",e[e.COMPRESSED_SEGMENTATION=2]="COMPRESSED_SEGMENTATION",e[e.COMPRESSO=3]="COMPRESSO",e[e.PNG=4]="PNG",e[e.JXL=5]="JXL",e))(jr||{}),xa=class{url;encoding;sharding;static RPC_ID="precomputed/VolumeChunkSource"},Sa=class{url;lod;static RPC_ID="precomputed/MeshSource"},ri=(e=>(e[e.RAW=0]="RAW",e[e.GZIP=1]="GZIP",e))(ri||{}),_a=(e=>(e[e.IDENTITY=0]="IDENTITY",e[e.MURMURHASH3_X86_128=1]="MURMURHASH3_X86_128",e))(_a||{});var wa=class{url;metadata;static RPC_ID="precomputed/MultiscaleMeshSource"},Ea=class{url;metadata;static RPC_ID="precomputed/SkeletonSource"},ba=class{url;sharding;static RPC_ID="precomputed/AnnotationSpatialIndexSource"},Ia=class{rank;relationships;properties;byId;type;static RPC_ID="precomputed/AnnotationSource"};var vv=Symbol("objectId"),KM=0;function xv(e){if(e instanceof Object){let t=e[vv];return t===void 0&&(t=e[vv]=KM++),`o${t}`}return""+JSON.stringify(e)}var ff=class extends fe{asyncMemoize;initialize(t){super.initialize(t)}freeSystemMemory(){this.asyncMemoize=void 0}},qe=class extends Hi{constructor(t,r){super(t),this.registerDisposer(t),this.downloadFunction=r.get,this.encodeKeyFunction=r.encodeKey??Kt}encodeKeyFunction;downloadFunction;get(t,r){let n=this.encodeKeyFunction(t),i=this.chunks.get(n);return i===void 0&&(i=this.getNewChunk_(ff),i.initialize(n),this.addChunk(i)),i.asyncMemoize===void 0&&(i.asyncMemoize=Yi(async s=>{try{let{data:o,size:a}=await this.downloadFunction(t,s);return i.systemMemoryBytes=a,i.queueManager.updateChunkState(i,z.SYSTEM_MEMORY_WORKER),o}catch(o){throw i.queueManager.updateChunkState(i,z.FAILED),o}})),i.state===z.SYSTEM_MEMORY_WORKER&&i.chunkManager.queueManager.markRecentlyUsed(i),i.asyncMemoize(r)}};function Sv(e,t,r){return e.memoize.get(`simpleAsyncCache:${t}`,()=>new qe(e.addRef(),r))}function ni(e,t,r,n){return e.chunkManager.memoize.get(`getCachedDecodedUrl:${xv(r)}`,()=>{let s=new qe(e.chunkManager.addRef(),{get:async(o,a)=>{let c=await e.kvStoreContext.read(o,{...a,throwIfMissing:!0});try{return r(c,a)}catch(l){throw new Error("Error reading ${url}",{cause:l})}}});return s.registerDisposer(e.addRef()),s}).get(t,n)}var YM=100,Gr=class{constructor(t,r){this.base=t,this.format=r}async stat(t){return await this.base.stat(t),{totalSize:void 0}}async read(t){let{byteRange:r}=t;if(r===void 0){let a=await this.base.read(t);return a===void 0?void 0:{response:new Response(Fu(a.response,this.format)),offset:0,length:void 0,totalSize:void 0}}if("suffixLength"in r||r.offset!==0)throw new Error(`Byte range with offset not supported: ${JSON.stringify(r)}`);let n=new Uint8Array(r.length),i=[],s=0,o=r.length+YM;for(;;){let a=await this.base.read({...t,byteRange:{offset:s,length:o-s}});if(a===void 0)return;{let p=new Uint8Array(await a.response.arrayBuffer());i.push(p),s+=p.length}let c=new DecompressionStream("gzip"),l=c.writable.getWriter(),u=[];for(let p of i)u.push(l.write(p));u.push(l.close());let f=c.readable.getReader(),h=0;try{for(;h<n.length;){let{value:p}=await f.read();if(p===void 0)break;let d=n.length-h;p.length>d&&(p=p.subarray(0,d)),n.set(p,h),h+=p.length}if(h===n.length||s===a.totalSize)return h<n.length&&(n=n.subarray(0,h)),{response:new Response(n),offset:0,length:n.length,totalSize:void 0}}finally{await f.cancel(),await Promise.allSettled(u)}o+=Math.min(100,n.length-h)}}getUrl(){return this.base.getUrl()+"|gzip"}};function Ca(e){return e^=e>>>16,e=Math.imul(e,2246822507),e^=e>>>13,e=Math.imul(e,3266489909),e^=e>>>16,e}function wv(e,t){return e<<t|e>>>32-t}function Ev(e,t){let r=e,n=e,i=e,s=e,o=597399067,a=2869860233,c=951274213,l=Math.imul(Number(t>>BigInt(32)),a);l=wv(l,16),l=Math.imul(l,c),n^=l;let u=Math.imul(Number(t&BigInt(4294967295)),o);u=wv(u,15),u=Math.imul(u,a),r^=u;let f=8;return r^=f,n^=f,i^=f,s^=f,r=r+n>>>0,r=r+i>>>0,r=r+s>>>0,n=n+r>>>0,i=i+r>>>0,s=s+r>>>0,r=Ca(r),n=Ca(n),i=Ca(i),s=Ca(s),r=r+n>>>0,r=r+i>>>0,r=r+s>>>0,n=n+r>>>0,BigInt(r)|BigInt(n)<<BigInt(32)}var JM=new Map([[_a.MURMURHASH3_X86_128,e=>Ev(0,e)],[_a.IDENTITY,e=>e]]);function bv(e,t){return t===ri.GZIP&&(e=new Gr(e,"gzip")),e}function HM(e,t,r){return new qe(e.addRef(),{encodeKey:n=>n.toString(),get:async(n,i)=>{let s=n&(1n<<BigInt(r.minishardBits))-1n,o=(1n<<BigInt(r.shardBits))-1n&n>>BigInt(r.minishardBits),a=t.path+o.toString(16).padStart(Math.ceil(r.shardBits/4),"0")+".shard",c=new ke(t.store,a),l=BigInt(16)<<BigInt(r.minishardBits),u=s<<4n,f=await Pr(c,{...i,byteRange:{offset:Number(u),length:16},strictByteRange:!0});if(f===void 0)return{data:void 0,size:0};let h=await f.response.arrayBuffer(),p=new DataView(h),d=p.getBigUint64(0,!0),m=p.getBigUint64(8,!0);if(d===m)return{data:void 0,size:0};d+=l,m+=l;let g=await(await Pr(bv(new tt(c,{offset:Number(d),length:Number(m-d)}),r.minishardIndexEncoding),{...i,strictByteRange:!0,throwIfMissing:!0})).response.arrayBuffer();if(g.byteLength%24!==0)throw new Error(`Invalid minishard index length: ${g.byteLength}`);let y=new BigUint64Array(g);u0(y,oe.LITTLE);let I=y.byteLength/24,_=0n,E=l;for(let b=0;b<I;++b){let C=_+y[b];_=y[b]=C;let v=E+y[I+b];y[I+b]=v;let w=y[2*I+b],x=v+w;E=x,y[2*I+b]=x}return{data:{data:y,shardPath:a},size:y.byteLength}}})}function WM(e,t){let r=e.data,n=r.length/3;for(let i=0;i<n;++i){if(r[i]!==t)continue;let s=r[n+i],o=r[2*n+i];return{offset:Number(s),length:Number(o-s)}}}var hf=class extends be{constructor(t,r,n){super(),this.base=r,this.sharding=n,this.minishardIndexCache=this.registerDisposer(HM(t,r,n))}minishardIndexCache;getUrl(t){return`chunk ${t} in ${this.base.store.getUrl(this.base.path)}`}async findKey(t,r){let{sharding:n}=this,o=JM.get(n.hash)(t>>BigInt(n.preshiftBits))&(1n<<BigInt(n.minishardBits+n.shardBits))-1n,a=await this.minishardIndexCache.get(o,r);if(a===void 0)return;let c=WM(a,t);if(c!==void 0)return{minishardEntry:c,shardInfo:{shardPath:a.shardPath,offset:c.offset}}}async readWithShardInfo(t,r){let{sharding:n}=this,i=await this.findKey(t,r);if(i===void 0)return;let{minishardEntry:s,shardInfo:o}=i;return{response:await bv(new tt(new ke(this.base.store,o.shardPath),s),n.dataEncoding).read(r),shardInfo:o}}async stat(t,r){let n=await this.findKey(t,r);if(n===void 0)return;let{sharding:i}=this;return i.dataEncoding!==ri.RAW?{totalSize:void 0}:{totalSize:n.minishardEntry.length}}async read(t,r){let n=await this.readWithShardInfo(t,r);if(n!==void 0)return n.response}get supportsOffsetReads(){return this.sharding.dataEncoding===ri.RAW}get supportsSuffixReads(){return this.sharding.dataEncoding===ri.RAW}};function dn(e,t,r){if(r!==void 0)return e.registerDisposer(new hf(e.chunkManager,t,r))}var fs,_v=0,Cv,Iv={emscripten_notify_memory_growth:e=>{},neuroglancer_draco_receive_decoded_mesh:(e,t,r,n,i)=>{let s=e*3,o=Cv.exports.memory,a=new Uint32Array(o.buffer,r,s).slice(),c=new Uint32Array(o.buffer,n,3*t).slice(),l=new Uint32Array(o.buffer,i,_v+1).slice();fs={indices:a,vertexPositions:c,subChunkOffsets:l}},proc_exit:e=>{throw`proc exit: ${e}`}},pf;function Av(){return pf==null&&(pf=(async()=>{let e=Cv=(await WebAssembly.instantiateStreaming(fetch(new URL("./neuroglancer_draco.wasm",import.meta.url)),{env:Iv,wasi_snapshot_preview1:Iv})).instance;return e.exports._initialize(),e})()),pf}async function Mv(e,t,r){let n=await Av(),i=n.exports.malloc(e.byteLength);new Uint8Array(n.exports.memory.buffer).set(e,i),_v=r?8:1;let o=n.exports.neuroglancer_draco_decode(i,e.byteLength,r,t,!0);if(o===0){let a=fs;if(fs=void 0,a instanceof Error)throw a;return a}throw new Error(`Failed to decode draco mesh: ${o}`)}async function kv(e){let t=await Av(),r=t.exports.malloc(e.byteLength);new Uint8Array(t.exports.memory.buffer).set(e,r);let i=t.exports.neuroglancer_draco_decode(r,e.byteLength,!1,0,!1);if(i===0){let s=fs;if(fs=void 0,s instanceof Error)throw s;return s.vertexPositions=new Float32Array(s.vertexPositions.buffer),s}throw new Error(`Failed to decode draco mesh: ${i}`)}function Tv(e,t,r){let n=new DataView(t),i=n.getUint32(0,!0),s=n.getUint32(4,!0),o=8,a=8+i*4*3;aa(e,t,oe.LITTLE,o,i,a,s),a+=s*4*2;let c=[];for(let l of r.values()){let u=lt[l.dataType]*l.numComponents,f=u*i,h=new Uint8Array(t,a,f);switch(u){case 2:l0(h,oe.LITTLE);break;case 4:case 8:It(h,oe.LITTLE);break}c.push(h),a+=f}e.vertexAttributes=c}var Nv=De("decodeCompresso");async function Dv(e,t,r){let n=await ue(Nv,t,[r],new Uint8Array(r));await ft(e,t,n.buffer)}var Pv=De("decodeJxl");async function Rv(e,t,r){let n=e.chunkDataSize,{uint8Array:i}=await ue(Pv,t,[r],new Uint8Array(r),n[0]*n[1]*n[2],n[3]||1,1);await ut(e,t,i)}async function Ov(e,t,r){let n=e.chunkDataSize,i=e.source.spec.dataType,{uint8Array:s}=await ue(hn,t,[r],new Uint8Array(r),void 0,void 0,n[0]*n[1]*n[2],n[3]||1,lt[i],!1);await ft(e,t,s.buffer)}var XM=Object.defineProperty,ZM=Object.getOwnPropertyDescriptor,ii=(e,t,r,n)=>{for(var i=n>1?void 0:n?ZM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&XM(t,r,i),i};var QM=!1;function df(e){if(e===void 0)throw new Error("not found");return e}var mn=new Map;mn.set(jr.RAW,ft);mn.set(jr.JPEG,Br);mn.set(jr.COMPRESSED_SEGMENTATION,Xn);mn.set(jr.COMPRESSO,Dv);mn.set(jr.PNG,Ov);mn.set(jr.JXL,Rv);var Uv=class extends ye(Te(Ae),xa){chunkDecoder=mn.get(this.parameters.encoding);kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.sharding);gridShape=(()=>{let e=new Uint32Array(3),{upperVoxelBound:t,chunkDataSize:r}=this.spec;for(let n=0;n<3;++n)e[n]=Math.ceil(t[n]/r[n]);return e})();async download(e,t){let{shardedKvStore:r}=this,n;if(r===void 0){let{kvStore:i}=this,s;{let o=this.computeChunkBounds(e),a=e.chunkDataSize;s=`${i.path}${o[0]}-${o[0]+a[0]}_${o[1]}-${o[1]+a[1]}_${o[2]}-${o[2]+a[2]}`}n=await i.store.read(s,{signal:t})}else{this.computeChunkBounds(e);let{gridShape:i}=this,{chunkGridPosition:s}=e,o=Math.ceil(Math.log2(i[0])),a=Math.ceil(Math.log2(i[1])),c=Math.ceil(Math.log2(i[2])),l=$o(o,a,c,s[0],s[1],s[2]);n=await r.read(l,{signal:t})}n!==void 0&&await this.chunkDecoder(e,t,await n.response.arrayBuffer())}};Uv=ii([j()],Uv);function Aa(e,t){return Yo(e,t,"fragments")}function ek(e,t){let n=new DataView(t).getUint32(0,!0);or(e,Yn(t,oe.LITTLE,4,n))}var Lv=class extends ye(Te(Nt),Sa){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);async download(e,t){let{parameters:r,kvStore:n}=this,i=await Jt(n.store,`${n.path}${e.objectId}:${r.lod}`,{signal:t,throwIfMissing:!0});Aa(e,await i.response.json())}async downloadFragment(e,t){let{kvStore:r}=this,n=await Jt(r.store,`${r.path}${e.fragmentId}`,{signal:t,throwIfMissing:!0});ek(e,await n.response.arrayBuffer())}};Lv=ii([j()],Lv);function tk(e,t){if(t.byteLength<28||t.byteLength%4!==0)throw new Error(`Invalid index file size: ${t.byteLength}`);let r=new DataView(t),n=0,i=B.fromValues(r.getFloat32(n,!0),r.getFloat32(n+4,!0),r.getFloat32(n+8,!0));n+=12;let s=B.fromValues(r.getFloat32(n,!0),r.getFloat32(n+4,!0),r.getFloat32(n+8,!0));n+=12;let o=r.getUint32(n,!0);if(n+=4,t.byteLength<n+(8+4*3)*o)throw new Error(`Invalid index file size for ${o} lods: ${t.byteLength}`);let a=new Float32Array(t,n,o);n+=4*o,It(a,oe.LITTLE);let c=new Float32Array(t,n,o*3);It(c,oe.LITTLE),n+=12*o;let l=new Uint32Array(t,n,o);n+=4*o,It(l,oe.LITTLE);let u=l.reduce((C,v)=>C+v);if(t.byteLength!==n+16*u)throw new Error(`Invalid index file size for ${o} lods and ${u} total fragments: ${t.byteLength}`);let f=new Uint32Array(t,n);It(f,oe.LITTLE);let h=B.fromValues(Number.POSITIVE_INFINITY,Number.POSITIVE_INFINITY,Number.POSITIVE_INFINITY),p=B.fromValues(Number.NEGATIVE_INFINITY,Number.NEGATIVE_INFINITY,Number.NEGATIVE_INFINITY),d=Math.max(1,a.length);{let C=0;for(let v=0;v<o;++v){let w=l[v];if(QM)for(let x=1;x<w;++x){let k=f[C+w*0+(x-1)],M=f[C+w*1+(x-1)],P=f[C+w*2+(x-1)],F=f[C+w*0+x],S=f[C+w*1+x],O=f[C+w*2+x];Gn(k,M,P,F,S,O)||console.log(`Fragment index violates zorder constraint: lod=${v}, chunk ${x-1} = [${k},${M},${P}], chunk ${x} = [${F},${S},${O}]`)}for(let x=0;x<3;++x){let k=Number.NEGATIVE_INFINITY,M=Number.POSITIVE_INFINITY,P=C+w*x;for(let F=0;F<w;++F){let S=f[P+F];k=Math.max(k,S),M=Math.min(M,S)}if(w!==0){for(;k>>>d-v-1!==M>>>d-v-1;)++d;v===0&&(h[x]=Math.min(h[x],(1<<v)*M),p[x]=Math.max(p[x],(1<<v)*(k+1)))}}C+=w*4}}let m=0;{let C=0,v=0;for(let w=0;w<o;++w){let x=l[w];m+=C*(w-v),v=w,C=x,m+=x}m+=(d-1-v)*C}let g=new Uint32Array(5*m),y=new Float64Array(m+1),I;{let C=0,v=0,w=0,x=0;for(let k=0;k<o;++k){let M=l[k];for(let P=0;P<M;++P){for(let S=0;S<3;++S)g[5*(v+P)+S]=f[x+P+S*M];let F=f[x+P+3*M];w+=F,y[v+P+1]=w,F===0&&(g[5*(v+P)+4]=2147483648)}for(x+=4*M,k!==0&&w0(g,C,v,v+M),C=v,v+=M;k+1<d&&(k+1>=a.length||a[k+1]===0);){let P=Ho(g,C,v);y.fill(w,v+1,P+1),C=v,v=P,++k}}I=g.slice(0,5*v),e.offsets=y.slice(0,v+1)}let _=e.source,{lodScaleMultiplier:E}=_.parameters.metadata,b=new Float32Array(d);b.set(a,0);for(let C=0;C<a.length;++C)b[C]*=E;e.manifest={chunkShape:i,chunkGridSpatialOrigin:s,clipLowerBound:B.add(h,s,B.multiply(h,h,i)),clipUpperBound:B.add(p,s,B.multiply(p,p,i)),octree:I,lodScales:b,vertexOffsets:c}}async function rk(e,t){let{lod:r}=e,n=e.manifestChunk.source,i=await Mv(new Uint8Array(t),n.parameters.metadata.vertexQuantizationBits,r!==0);Jo(e,i,n.format.vertexPositionFormat)}var Fv=class extends ye(Te(Kn),wa){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.metadata.sharding);async download(e,t){let{shardedKvStore:r}=this,n;if(r===void 0){let{kvStore:s}=this;n=await s.store.read(`${s.path}${e.objectId}.index`,{signal:t})}else({response:n,shardInfo:e.shardInfo}=df(await r.readWithShardInfo(e.objectId,{signal:t})));let i=await df(n).response.arrayBuffer();tk(e,i)}async downloadFragment(e,t){let{kvStore:r}=this,n=e.manifestChunk,i=e.chunkIndex,{shardInfo:s,offsets:o}=n,a=o[i],c=o[i+1],l,u,f;if(s!==void 0){l=s.shardPath;let p=o[o.length-1],d=s.offset-p+a,m=d+c-a;u=d,f=m}else l=`${r.path}${n.objectId}`,u=a,f=c;let h=await Jt(r.store,l,{signal:t,byteRange:{offset:u,length:f-u},throwIfMissing:!0,strictByteRange:!0});await rk(e,await h.response.arrayBuffer())}};Fv=ii([j()],Fv);async function mf(e,t,r){let{shardedKvStore:n}=e;if(n===void 0){let{kvStore:i}=e;return i.store.read(`${i.path}${t}`,{signal:r})}else return n.read(t,{signal:r})}var Bv=class extends ye(Te($r),Ea){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.metadata.sharding);async download(e,t){let{parameters:r}=this,n=df(await mf(this,e.objectId,t));Tv(e,await n.response.arrayBuffer(),r.metadata.vertexAttributes)}};Bv=ii([j()],Bv);function Vv(e,t,r){if(e.byteLength<8)throw new Error("Expected at least 8 bytes");let i=new DataView(e);if(i.getUint32(4,!0)!==0)throw new Error("Annotation count too high");let o=i.getUint32(0,!0),a=r.serializedBytes,c=8+a*o,l=o,u=t.type;if(u===Oe.POLYLINE){let E=sk(i,t.rank,o,a,!0);l=E.numInstances,c=E.totalBytes}let f=ok(e,i,o,c),h=new Uint8Array(e,8,c-8),p=new Vn,d=p.typeToInstanceCounts=new Array(br.length);d.fill([]);let{propertyGroupBytes:m}=r;if(m.length>1||u===Oe.POLYLINE){let E=nk(i,h,r,l,o,u,t.rank,!0);p.data=E.outputData,d[Oe.POLYLINE]=E.polylineInstanceCounts}else p.data=h,d[t.type]=Array.from({length:f.length},(E,b)=>b);(p.typeToOffset=new Array(br.length)).fill(0);let y=p.typeToIds=new Array(br.length),I=p.typeToIdMaps=new Array(br.length),_=p.typeToSize=new Array(br.length);return _.fill(0),_[t.type]=l,y.fill([]),y[t.type]=f,I.fill(new Map),I[t.type]=new Map(f.map((E,b)=>[E,b])),p}function nk(e,t,r,n,i,s,o,a){let{propertyGroupBytes:c,serializedBytes:l}=r,u=l*n,f=new Uint8Array(u),h=[];s===Oe.POLYLINE&&(h=ik(e,t,f,r,i,o,a));let p=t;if(s===Oe.POLYLINE&&(p=new Uint8Array(f)),c.length>1){let d=0,m=0;for(let g=0;g<c.length;++g){let y=0,I=c[g];for(let _=0;_<i;++_){let E=1;s===Oe.POLYLINE&&(_===i-1?E=n-h[_]:E=h[_+1]-h[_]);for(let b=0;b<E;++b){let C=d+y*l,v=m+y*I;f.set(p.subarray(C,C+I),v),++y}}d+=I,m+=I*n}}return{outputData:f,polylineInstanceCounts:h}}function ik(e,t,r,n,i,s,o){let a=new DataView(r.buffer),c=0,l=0,u=4,f=s*4,h=n.serializedBytes,p=h-2*f-u,d=8,m=0,g=new Array(i);for(let y=0;y<i;++y){let I=e.getUint32(c+d,o),_=I-1;c+=u;let E=c+I*f;g[y]=m,m+=_;for(let b=0;b<_;++b){let C=b===_-1?1:0,v=b|C<<31;a.setUint32(l,v,o),r.set(t.subarray(c,c+2*f),l+u),r.set(t.subarray(E,E+p),l+u+2*f),c+=f,l+=h}c=E+p}return g}function sk(e,t,r,n,i){let s=8,o=0;for(let a=0;a<r;a++){let c=e.getUint32(s,i),l=c-1,u=c*t*4,f=n-2*t*4;s+=u+f,o+=l}return{totalBytes:s,numInstances:o}}function ok(e,t,r,n){let i=n+8*r;if(e.byteLength!==i)throw new Error(`Expected ${i} bytes, but received: ${e.byteLength} bytes`);let s=n,o=new Array(r);for(let a=0;a<r;++a)o[a]=t.getBigUint64(s+a*8,!0).toString();return o}function ak(e,t,r,n){let i=as[t.type],s=r.serializedBytes,o=new DataView(e),a=0;if(t.type===Oe.POLYLINE){let h=o.getUint32(0,!0)&2147483647,p=r.serializedBytes-(2*4*t.rank+4);s=4+h*4*t.rank+p,a=(h-2)*4*t.rank}let c=t.relationships.length,l=s+4*c;if(e.byteLength<l)throw new Error(`Expected at least ${l} bytes, but received: ${e.byteLength}`);let u=i.deserialize(o,0,!0,t.rank,n,0);r.deserialize(o,a,0,1,!0,u.properties=new Array(t.properties.length)),a=s;let f=u.relatedSegments=[];f.length=c;for(let h=0;h<c;++h){let p=o.getUint32(a,!0);if(e.byteLength<l+p*8)throw new Error(`Expected at least ${l} bytes, but received: ${e.byteLength}`);a+=4;let d=f[h]=new BigUint64Array(p);for(let m=0;m<p;++m)d[m]=o.getBigUint64(a,!0),a+=8}if(a!==e.byteLength)throw new Error(`Expected ${a} bytes, but received: ${e.byteLength}`);return u}var zv=class extends ye(Te(cn),ba){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.sharding);async download(e,t){let{shardedKvStore:r}=this,{parent:n}=this,i,{chunkGridPosition:s}=e;if(r===void 0){let{kvStore:o}=this,a=`${o.path}${s.join("_")}`;i=await o.store.read(a,{signal:t})}else{let{upperChunkBound:o}=this.spec,{chunkGridPosition:a}=e,c=r0(a,o);i=await r.read(c,{signal:t})}i!==void 0&&(e.data=Vv(await i.response.arrayBuffer(),n.parameters,n.annotationPropertySerializer))}};zv=ii([j()],zv);var $v=class extends ye(Te(jn),Ia){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.byId.url);shardedKvStore=dn(this,this.kvStore,this.parameters.byId.sharding);relationshipIndexSource=this.parameters.relationships.map(e=>{let t=this.sharedKvStoreContext.kvStoreContext.getKvStore(e.url),r=dn(this,t,e.sharding);return{kvStore:t,shardedKvStore:r}});annotationPropertySerializer=new os(this.parameters.rank,as[this.parameters.type].serializedBytes(this.parameters.rank),this.parameters.properties);async downloadSegmentFilteredGeometry(e,t,r){let n=await mf(this.relationshipIndexSource[t],e.objectId,r);n!==void 0&&(e.data=Vv(await n.response.arrayBuffer(),this.parameters,this.annotationPropertySerializer))}async downloadMetadata(e,t){let r=BigInt(e.key),n=await mf(this,r,t);n===void 0?e.annotation=null:e.annotation=ak(await n.response.arrayBuffer(),this.parameters,this.annotationPropertySerializer,e.key)}};$v=ii([j()],$v);var ck=Object.defineProperty,lk=Object.getOwnPropertyDescriptor,yf=(e,t,r,n)=>{for(var i=n>1?void 0:n?lk(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&ck(t,r,i),i};function uk(e,t,r){if(t&&t.charAt(0)==="~"){let n=t.substring(1).split(":"),i={offset:Number(n[1]),length:Number(n[2])};return Jt(e.store,`${e.path}initial/${n[0]}`,{signal:r,byteRange:i,throwIfMissing:!0})}return Jt(e.store,`${e.path}dynamic/${t}`,{signal:r,throwIfMissing:!0})}function fk(e,t,r,n){return r.sharding?uk(e,t,n):Jt(e.store,`${e.path}/${t}`,{signal:n,throwIfMissing:!0})}async function hk(e,t){let r=await kv(t);or(e,r)}var jv=class extends ye(Te(Nt),va){manifestRequestCount=new Map;newSegments=new $n;manifestHttpSource=uf(this.sharedKvStoreContext.kvStoreContext,this.parameters.manifestUrl);fragmentKvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.fragmentUrl);addNewSegment(e){let{newSegments:t}=this;t.add(e);let r=1e3*60*10;setTimeout(()=>{t.delete(e)},r)}async download(e,t){let{parameters:r,newSegments:n,manifestRequestCount:i}=this;if(lf(e.objectId,r.nBitsForLayerId))return Aa(e,{fragments:[]});let{fetchOkImpl:s,baseUrl:o}=this.manifestHttpSource,a=`/manifest/${e.objectId}:${r.lod}?verify=1&prepend_seg_ids=1`,c=await(await s(o+a,{signal:t})).json(),l=a;if(n.has(e.objectId)){let u=(i.get(l)??0)+1;i.set(l,u),setTimeout(()=>{this.chunkManager.queueManager.updateChunkState(e,z.QUEUED)},2**u*1e3)}else i.delete(l);return Aa(e,c)}async downloadFragment(e,t){let{response:r}=await fk(this.fragmentKvStore,e.fragmentId,this.parameters,t);await hk(e,new Uint8Array(await r.arrayBuffer()))}getFragmentKey(e,t){return pv(t)}};jv=yf([j()],jv);var gf=class extends fe{chunkGridPosition;source=null;segment;leaves=new BigUint64Array(0);chunkDataSize;initializeVolumeChunk(t,r){super.initialize(t),this.chunkGridPosition=Float32Array.from(r)}initializeChunkedGraphChunk(t,r,n){this.initializeVolumeChunk(t,r),this.chunkDataSize=null,this.systemMemoryBytes=16,this.gpuMemoryBytes=0,this.segment=n}downloadSucceeded(){this.systemMemoryBytes=16,this.systemMemoryBytes+=this.leaves.byteLength,this.queueManager.updateChunkState(this,z.SYSTEM_MEMORY_WORKER),this.priorityTier<se.RECENT&&this.source.chunkManager.scheduleUpdateChunkPriorities(),super.downloadSucceeded()}freeSystemMemory(){this.leaves=new BigUint64Array(0)}};function pk(e){return BigUint64Array.from(e,yr)}var Gv=class extends ye(Te(Ne),ya){spec;tempChunkDataSize;tempChunkPosition;httpSource=uf(this.sharedKvStoreContext.kvStoreContext,this.parameters.url);constructor(e,t){super(e,t),this.spec=t.spec;let r=this.spec.rank;this.tempChunkDataSize=new Uint32Array(r),this.tempChunkPosition=new Float32Array(r)}async download(e,t){let r=this.computeChunkBounds(e),n=e.chunkDataSize,i=`${r[0]}-${r[0]+n[0]}_${r[1]}-${r[1]+n[1]}_${r[2]}-${r[2]+n[2]}`,{fetchOkImpl:s,baseUrl:o}=this.httpSource,a=s(`${o}/${e.segment}/leaves?int64_as_str=1&bounds=${i}`,{signal:t});await this.withErrorMessage(a,`Fetching leaves of segment ${e.segment} in region ${i}: `).then(c=>c.json()).then(c=>{e.leaves=pk(c.leaf_ids)}).catch(c=>{c instanceof Error&&c.name==="AbortError"||console.error(c)})}getChunk(e,t){let r=`${tr(e)}-${t}`,n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(gf),n.initializeChunkedGraphChunk(r,e,t),this.addChunk(n)),n}computeChunkBounds(e){return Vu(this,e)}async withErrorMessage(e,t){return e.catch(async r=>{if(r instanceof bt&&r.response){let n=await yv(r);throw new Error(`[${r.response.status}] ${t}${n??""}`)}throw r})}};Gv=yf([j()],Gv);var qv=B.create(),dk=B.create(),mk=B.create(),Kv=class extends an(Et(Xe(ir))){source;localPosition;leafRequestsActive;nBitsForLayerId;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.localPosition=e.get(t.localPosition),this.leafRequestsActive=e.get(t.leafRequestsActive),this.nBitsForLayerId=e.get(t.nBitsForLayerId),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities(),this.debouncedupdateDisplayState()}))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(t),e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.state={displayDimensionRenderInfo:r.projectionParameters.value.displayDimensionRenderInfo}}get renderRatioLimit(){return gv}updateChunkPriorities(){let{source:e,chunkManager:t}=this;t.registerLayer(this);for(let r of this.attachments.values()){let{view:n}=r,i=n.visibility.value;if(i===Number.NEGATIVE_INFINITY)continue;let s=r.state,{transformedSource:o}=s,a=n.projectionParameters.value;if(!o)continue;let c=a.pixelSize*1.1,l=o.effectiveVoxelSize;if(this.leafRequestsActive.value=this.renderRatioLimit>=c/Math.min(...l),!this.leafRequestsActive.value)continue;let u=je(i),f=Ge(i),{chunkLayout:h}=o,{size:p,finiteRank:d}=h,m=mk,g=dk;B.copy(m,p);for(let I=d;I<3;++I)m[I]=0,g[I]=0;let{centerDataPosition:y}=a;h.globalToLocalSpatial(g,y),ko(a,this.localPosition.value,o,To(a,h),I=>{B.multiply(qv,I,m);let _=-B.distance(g,qv),{curPositionInChunks:E}=o;xr(this,(b,C)=>{if(lf(b,this.nBitsForLayerId.value))return;let v=e.getChunk(E,b);t.requestChunk(v,u,f+_,z.SYSTEM_MEMORY_WORKER),++this.numVisibleChunksNeeded,v.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable})})}}forEachSelectedRootWithLeaves(e){let{source:t}=this;for(let r of t.chunks.values())r.state===z.SYSTEM_MEMORY_WORKER&&r.priorityTier<se.RECENT&&this.visibleSegments.has(r.segment)&&r.leaves.length&&e(r.segment,r.leaves)}debouncedupdateDisplayState=tn(()=>{this.updateDisplayState()},100);updateDisplayState(){let e=new Map,t=new Map;this.forEachSelectedRootWithLeaves((r,n)=>{t.set(r,(t.get(r)??0)+n.length)}),this.forEachSelectedRootWithLeaves((r,n)=>{e.has(r)||(e.set(r,new $n),e.get(r).reserve(t.get(r)),e.get(r).add(r)),e.get(r).add(n)});for(let[r,n]of e){let i=[...n].filter(s=>!this.segmentEquivalences.has(s));for(let s of i)this.segmentEquivalences.link(r,s)}}};Kv=yf([j(dv)],Kv);X(mv,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);n.state.transformedSource=sn(this,e.sources,r)[0][0],n.state.displayDimensionRenderInfo=e.displayDimensionRenderInfo,r.chunkManager.scheduleUpdateChunkPriorities()});X(hv,function(e){this.get(e.rpcId).addNewSegment(e.segment)});var Ma=De("decodeBlosc");var qr=De("decodeZstd");var si=(e=>(e[e.RAW=0]="RAW",e[e.ZLIB=1]="ZLIB",e[e.GZIP=2]="GZIP",e[e.BLOSC=3]="BLOSC",e[e.ZSTD=4]="ZSTD",e))(si||{}),ka=class{url;encoding;static RPC_ID="n5/VolumeChunkSource"};var gk=Object.defineProperty,yk=Object.getOwnPropertyDescriptor,vk=(e,t,r,n)=>{for(var i=n>1?void 0:n?yk(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&gk(t,r,i),i};async function xk(e,t,r,n){let i=new DataView(r),s=i.getUint16(0,!1);if(s!==0)throw new Error(`Unsupported mode: ${s}.`);let o=i.getUint16(2,!1);if(o!==e.source.spec.rank)throw new Error("Number of dimensions must be 3.");let a=4,c=new Uint32Array(o);for(let u=0;u<o;++u)c[u]=i.getUint32(a,!1),a+=4;e.chunkDataSize=c;let l=new Uint8Array(r,a);switch(n){case si.ZLIB:l=new Uint8Array(await wr(l,"deflate"));break;case si.GZIP:l=new Uint8Array(await wr(l,"gzip"));break;case si.BLOSC:l=await ue(Ma,t,[l.buffer],l);break;case si.ZSTD:l=await ue(qr,t,[l.buffer],l);break}await ft(e,t,l.buffer,oe.BIG,l.byteOffset,l.byteLength)}var Yv=class extends ye(Te(Ae),ka){chunkKvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);async download(e,t){let{parameters:r,chunkKvStore:n}=this,{chunkGridPosition:i}=e,s=n.path,o=this.spec.rank;for(let c=0;c<o;++c)c!==0&&(s+="/"),s+=`${i[c]}`;let a=await n.store.read(s,{signal:t});a!==void 0&&await xk(e,t,await a.response.arrayBuffer(),r.encoding)}};Yv=vk([j()],Yv);var Ke=Ri(Rx(),1);var Ox="nifti/getNiftiVolumeInfo",Xa=class{url;static RPC_ID="nifti/VolumeChunkSource"};var Qk=Object.defineProperty,eT=Object.getOwnPropertyDescriptor,tT=(e,t,r,n)=>{for(var i=n>1?void 0:n?eT(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&Qk(t,r,i),i};var Kf=class{uncompressedData;header};async function rT(e,t){let r=await e.response.arrayBuffer();(0,Ke.isCompressed)(r)&&(r=await wr(r,"gzip",t.signal));let n=new Kf;n.uncompressedData=r;let i=(0,Ke.readHeader)(r);if(i===null)throw new Error("Failed to parse NIFTI header.");return n.header=i,{data:n,size:r.byteLength}}function Lx(e,t,r){return ni(e,t,rT,r)}async function nT(e,t,r){return(await Lx(e,t,r)).header}function iT(e){return ge.fromValues(e[0][0],e[1][0],e[2][0],e[3][0],e[0][1],e[1][1],e[2][1],e[3][1],e[0][2],e[1][2],e[2][2],e[3][2],e[0][3],e[1][3],e[2][3],e[3][3])}var Fx=(e=>(e[e.NONE=0]="NONE",e[e.BINARY=1]="BINARY",e[e.UINT8=2]="UINT8",e[e.INT16=4]="INT16",e[e.INT32=8]="INT32",e[e.FLOAT32=16]="FLOAT32",e[e.COMPLEX64=32]="COMPLEX64",e[e.FLOAT64=64]="FLOAT64",e[e.RGB24=128]="RGB24",e[e.INT8=256]="INT8",e[e.UINT16=512]="UINT16",e[e.UINT32=768]="UINT32",e[e.INT64=1024]="INT64",e[e.UINT64=1280]="UINT64",e[e.FLOAT128=1536]="FLOAT128",e[e.COMPLEX128=1792]="COMPLEX128",e[e.COMPLEX256=2048]="COMPLEX256",e))(Fx||{}),sT=new Map([[256,{dataType:W.INT8}],[2,{dataType:W.UINT8}],[4,{dataType:W.INT16}],[512,{dataType:W.UINT16}],[8,{dataType:W.INT32}],[768,{dataType:W.UINT32}],[1024,{dataType:W.UINT64}],[1280,{dataType:W.UINT64}],[16,{dataType:W.FLOAT32}]]);xt(Ox,async function(e,t){let r=this.get(e.sharedKvStoreContext),n=await nT(r,e.url,t),i=sT.get(n.datatypeCode);if(i===void 0)throw new Error(`Unsupported data type: ${Fx[n.datatypeCode]||n.datatypeCode}.`);let s=1,o="";switch(n.xyzt_units&Ke.NIFTI1.SPATIAL_UNITS_MASK){case Ke.NIFTI1.UNITS_METER:s=1,o="m";break;case Ke.NIFTI1.UNITS_MM:s=1e3,o="m";break;case Ke.NIFTI1.UNITS_MICRON:s=1e6,o="m";break}let a="",c=1;switch(n.xyzt_units&Ke.NIFTI1.TEMPORAL_UNITS_MASK){case Ke.NIFTI1.UNITS_SEC:a="s",c=1;break;case Ke.NIFTI1.UNITS_MSEC:a="s",c=1e3;break;case Ke.NIFTI1.UNITS_USEC:a="s",c=1e6;break;case Ke.NIFTI1.UNITS_HZ:a="Hz",c=1;break;case Ke.NIFTI1.UNITS_RADS:a="rad/s",c=1;break}let l=[o,o,o,a,"","",""],u=Float64Array.of(n.pixDims[1]/s,n.pixDims[2]/s,n.pixDims[3]/s,n.pixDims[4]/c,n.pixDims[5],n.pixDims[6],n.pixDims[7]),f=Float64Array.of(1/s,1/s,1/s,1/c,1,1,1),h=["i","j","k","m","c^","c1^","c2^"],p=["x","y","z","t","c^","c1^","c2^"],d=n.dims[0];h=h.slice(0,d),p=p.slice(0,d),l=l.slice(0,d),u=u.slice(0,d),f=f.slice(0,d);let{quatern_b:m,quatern_c:g,quatern_d:y}=n,I=Math.sqrt(1-m*m-g*g-y*y),_=n.pixDims[0]===-1?-1:1,E=B.fromValues(n.qoffset_x,n.qoffset_y,n.qoffset_z),b=iT(n.affine),C=ig(ge.create(),E,Qt.fromValues(m,g,y,I),eg,_),v=nu(Float64Array,d+1),w=Math.min(3,d);for(let k=0;k<w;++k){for(let M=0;M<w;++M)v[M*(d+1)+k]=C[M*4+k];v[d*(d+1)+k]=C[12+k]}return{value:{rank:d,sourceNames:h,viewNames:p,units:l,sourceScales:u,viewScales:f,description:n.description,transform:v,dataType:i.dataType,volumeSize:Uint32Array.from(n.dims.slice(1,1+d))}}});var Ux=class extends ye(Te(Ae),Xa){async download(e,t){e.chunkDataSize=this.spec.chunkDataSize;let r=await Lx(this.sharedKvStoreContext,this.parameters.url,{signal:t}),n=(0,Ke.readImage)(r.header,r.uncompressedData);await ft(e,t,n,r.header.littleEndian?oe.LITTLE:oe.BIG)}};Ux=tT([j()],Ux);var Bx=De("parseOBJFromArrayBuffer");var zx="single_mesh/SingleMeshLayer",$x="single_mesh/getSingleMeshInfo",Vx="",Yf=class{meshSourceUrl},Za=class extends Yf{info;static RPC_ID="single_mesh/SingleMeshSource"};var oT=Object.defineProperty,aT=Object.getOwnPropertyDescriptor,qx=(e,t,r,n)=>{for(var i=n>1?void 0:n?aT(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&oT(t,r,i),i};var cT=50,Jf=class extends fe{data=null;freeSystemMemory(){this.data=null}serialize(t,r){super.serialize(t,r);let{vertexPositions:n,indices:i,vertexNormals:s,vertexAttributes:o}=this.data;t.vertexPositions=n,t.indices=i,t.vertexNormals=s,t.vertexAttributes=o;let a=new Set;a.add(n.buffer),a.add(i.buffer),a.add(s.buffer);for(let c of o)a.add(c.buffer);r.push(...a),this.data=null}downloadSucceeded(){let{vertexPositions:t,indices:r,vertexNormals:n,vertexAttributes:i}=this.data,s=this.gpuMemoryBytes=t.byteLength+r.byteLength+n.byteLength;for(let o of i)s+=o.byteLength;this.systemMemoryBytes=this.gpuMemoryBytes=s,super.downloadSucceeded()}},Kx=new Map;function Qa(e,t){Kx.set(e,t)}var lT=/^(?:([a-zA-Z-+_]+):\/\/)?(.*)$/;function uT(e,t){let r=t.match(lT);if(r===null||r[1]===void 0)throw new Error('Data source URL must have the form "<protocol>://<path>".');let n=r[1],i=e.get(n);if(i===void 0)throw new Error(`Unsupported data source: ${JSON.stringify(n)}.`);return[i,r[2],n]}function fT(e,t,r){let[n,i]=uT(Kx,t);return n.getMesh(e,i,r)}function Yx(e,t,r){return fT(e,t.meshSourceUrl,r)}var jx=class extends ye(Te(Ne),Za){getChunk(){let e=Vx,t=this.chunks.get(e);return t===void 0&&(t=this.getNewChunk_(Jf),t.initialize(e),this.addChunk(t)),t}async download(e,t){let r=await Yx(this.sharedKvStoreContext,this.parameters,{signal:t});if(Kt(r.info)!==Kt(this.parameters.info))throw new Error("Mesh info has changed.");r.vertexNormals===void 0&&(r.vertexNormals=Ru(r.vertexPositions,r.indices)),e.data=r}};jx=qx([j()],jx);var hT=Et(Xe(me)),Gx=class extends hT{source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;let t=je(e),r=Ge(e),{source:n,chunkManager:i}=this,s=n.getChunk();i.requestChunk(s,t,r+cT)}};Gx=qx([j(zx)],Gx);xt($x,async function(e,t){let r=this.get(e.sharedKvStoreContext),n=e.parameters;return{value:(await Yx(r,n,t)).info}});async function pT(e,t){let r=await e.response.arrayBuffer();return ue(Bx,t.signal,[r],r)}Qa("obj",{description:"OBJ",getMesh:(e,t,r)=>ni(e,t,pT,r)});var Ss=!1;var Hf=class{baseUrl;owner;project;stack;channel},Wf=class extends Hf{renderArgs},ec=class extends Wf{dims;level;encoding;static RPC_ID="render/TileChunkSource"};var dT=Object.defineProperty,mT=Object.getOwnPropertyDescriptor,gT=(e,t,r,n)=>{for(var i=n>1?void 0:n?mT(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&dT(t,r,i),i};var ws=new Map;ws.set("jpg",async(e,t,r)=>{let n=e.chunkDataSize,{uint8Array:i}=await ue(Hn,t,[r],new Uint8Array(r),void 0,void 0,n[0]*n[1]*n[2],3,!0);await ut(e,t,i)});ws.set("png",async(e,t,r)=>{let n=e.chunkDataSize,{uint8Array:i}=await ue(hn,t,[r],new Uint8Array(r),n[0],n[1],n[0]*n[1]*n[2],4,1,!1);await ut(e,t,i)});ws.set("png16",async(e,t,r)=>{let n=e.chunkDataSize,{uint8Array:i}=await ue(hn,t,[r],new Uint8Array(r),n[0],n[1],n[0]*n[1]*n[2],1,2,!1);await ut(e,t,i)});ws.set("raw16",(e,t,r)=>ft(e,t,r,oe.BIG));var Jx=class extends ye(Ae,ec){chunkDecoder=ws.get(this.parameters.encoding);queryString=(()=>{let{parameters:e}=this,t=new URLSearchParams;e.channel!==void 0&&t.append("channel",e.channel);for(let[r,n]of Object.entries(e.renderArgs))t.append(r,n);return t.toString()})();async download(e,t){let{parameters:r}=this,{chunkGridPosition:n}=e,i=1/2**r.level;e.chunkDataSize=this.spec.chunkDataSize;let s=e.chunkDataSize[0]*2**r.level,o=e.chunkDataSize[1]*2**r.level,a=B.create();a[0]=n[0]*s,a[1]=n[1]*o,a[2]=n[2];let c;r.encoding==="raw16"?c="raw16-image":r.encoding==="png16"?c="png16-image":r.encoding==="png"?c="png-image":c="jpeg-image";let l=`/render-ws/v1/owner/${r.owner}/project/${r.project}/stack/${r.stack}/z/${a[2]}/box/${a[0]},${a[1]},${s},${o},${i}/${c}`,u=await Re(`${r.baseUrl}${l}?${this.queryString}`,{signal:t});await this.chunkDecoder(e,t,await u.arrayBuffer())}};Jx=gT([j()],Jx);var Hx=De("parseVTKFromArrayBuffer");async function yT(e,t){let r=await e.response.arrayBuffer();return ue(Hx,t.signal,[r],r)}Qa("vtk",{description:"VTK",getMesh:async(e,t,r)=>{let n=await ni(e,t,yT,r),i={info:{numTriangles:n.numTriangles,numVertices:n.numVertices,vertexAttributes:[]},indices:n.indices,vertexPositions:n.vertexPositions,vertexAttributes:[]};for(let s of n.vertexAttributes)i.info.vertexAttributes.push({name:s.name,dataType:W.FLOAT32,numComponents:s.numComponents}),i.vertexAttributes.push(s.data);return i}});var ne=(e=>(e[e.arrayToArray=0]="arrayToArray",e[e.arrayToBytes=1]="arrayToBytes",e[e.bytesToBytes=2]="bytesToBytes",e))(ne||{});var xi={[ne.arrayToArray]:new Map,[ne.arrayToBytes]:new Map,[ne.bytesToBytes]:new Map,sharding:new Map};function pt(e){e.kind===ne.arrayToBytes&&"getShardedKvStore"in e?xi.sharding.set(e.name,e):xi[e.kind].set(e.name,e)}async function tc(e,t,r){let n=e[ne.bytesToBytes];for(let o=n.length;o--;){let a=n[o],c=xi[ne.bytesToBytes].get(a.name);if(c===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(a.name)}`);t=await c.decode(a.configuration,t,r)}let i;{let o=e[ne.arrayToBytes],a=xi[ne.arrayToBytes].get(o.name);if(a===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(o.name)}`);i=await a.decode(o.configuration,e.arrayInfo[e.arrayInfo.length-1],t,r)}let s=e[ne.arrayToArray];for(let o=s.length;o--;){let a=s[o],c=xi[ne.arrayToArray].get(a.name);if(c===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(a.name)}`);i=await c.decode(a.configuration,e.arrayInfo[o],i,r)}return i}function Wx(e,t,r){let n=r.store,i=t;for(;;){let{shardingInfo:c}=i;if(c===void 0)break;let l=i[ne.arrayToBytes],u=xi.sharding.get(l.name);if(u===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(l.name)}`);n=u.getShardedKvStore(l.configuration,e,n),i=c.subChunkCodecs}let s=i,o=r.path;function a(c,l){let u=o+l,f=c.length,h=t;for(;h.shardingInfo!==void 0;){let p=t.layoutInfo[t.layoutInfo.length-1],{physicalToLogicalDimension:d,readChunkShape:m}=p,{subChunkShape:g,subChunkGridShape:y,subChunkCodecs:I}=h.shardingInfo,_=new Array(f);for(let E=0;E<f;++E){let b=d[f-1-E];_[b]=Math.floor(c[E]*m[b]/g[b])%y[b]}u={base:u,subChunk:_},h=I}return u}return{kvStore:n,getChunkKey:a,decodeCodecs:s}}pt({name:"blosc",kind:ne.bytesToBytes,decode(e,t,r){return ue(Ma,r,[t.buffer],t)}});pt({name:"zstd",kind:ne.bytesToBytes,decode(e,t,r){return ue(qr,r,[t.buffer],t)}});pt({name:"bytes",kind:ne.arrayToBytes,async decode(e,t,r,n){let{dataType:i,chunkShape:s}=t,o=s.reduce((u,f)=>u*f,1),a=lt[i],c=o*a;if(r.byteLength!==c)throw new Error(`Raw-format chunk is ${r.byteLength} bytes, but ${o} * ${a} = ${c} bytes are expected.`);let l=Co(i,r.buffer,r.byteOffset,r.byteLength);return qn(l,e.endian,a),l}});var Xf=4;pt({name:"crc32c",kind:ne.bytesToBytes,async decode(e,t,r){if(t.length<Xf)throw new Error(`Expected buffer of size at least ${Xf} bytes but received: ${t.length} bytes`);return t.subarray(0,t.length-Xf)}});var rc=class{url;metadata;static RPC_ID="zarr/VolumeChunkSource"};for(let[e,t]of[["gzip","gzip"],["zlib","deflate"]])pt({name:e,kind:ne.bytesToBytes,async decode(r,n,i){return new Uint8Array(await wr(n,t,i))}});function Zf(e,t,r){Ee(e);let n=re(e,"name",s=>t(et(s))),i=re(e,"configuration",s=>(s===void 0?s={}:Ee(s),r(s,n)));return{name:n,configuration:i}}function vT(e){let{name:t,configuration:r}=Zf(e,n=>{let i=Xx.get(n);if(i===void 0)throw new Error(`Unknown codec: ${JSON.stringify(n)}`);return i},n=>n);return{resolver:t,configuration:r}}var Xx=new Map;function Zx(e){Xx.set(e.name,e)}function nc(e,t){let r=[],n=[],i=[],s=[];n.push(t);let o=ct(e,vT),a=o.length,c=0;for(;c<a;++c){let{resolver:m,configuration:g}=o[c];if(m.kind!==ne.arrayToArray)break;let y=m,{configuration:I,encodedArrayInfo:_}=y.resolve(g,t);n.push(_),t=_,r.push({kind:ne.arrayToArray,name:m.name,configuration:I})}if(c===a||o[c].resolver.kind!==ne.arrayToBytes)throw new Error("Missing array -> bytes codec");let{codecSpec:l,layoutInfo:u,encodedSize:f,shardingInfo:h}=(()=>{let{resolver:m,configuration:g}=o[c],y=m,{configuration:I,shardingInfo:_,encodedSize:E}=y.resolve(g,t);if(_!==void 0&&c+1!==a)throw new Error("bytes -> bytes codecs not supported following sharding codec");let b=y.getDecodedArrayLayoutInfo(I,t);return{codecSpec:{name:m.name,kind:ne.arrayToBytes,configuration:I},layoutInfo:b,encodedSize:E,shardingInfo:_}})();i[c]=u,s.push(f);let p=f,d=[];for(++c;c<a;){let{resolver:m,configuration:g}=o[c];if(m.kind!==ne.bytesToBytes)throw new Error(`Expected bytes -> bytes codec, but received ${JSON.stringify(m.name)} of kind ${ne[m.kind]}`);let y=m,{configuration:I,encodedSize:_}=y.resolve(g,p);d.push({name:m.name,kind:m.kind,configuration:I}),s.push(_),++c}for(let m=r.length-1;m>=0;--m)i[m]=o[m].resolver.getDecodedArrayLayoutInfo(r[m].configuration,n[m],i[m+1]);return{[ne.arrayToArray]:r,[ne.arrayToBytes]:l,[ne.bytesToBytes]:d,arrayInfo:n,layoutInfo:i,shardingInfo:h,encodedSize:s}}var ic=(e=>(e[e.DEFAULT=0]="DEFAULT",e[e.V2=1]="V2",e))(ic||{});function e1(e,t){return We(new Array(t),e,r=>{if(typeof r!="number"||!Number.isInteger(r)||r<=0)throw new Error(`Expected positive integer, but received: ${JSON.stringify(r)}`);return r})}var Qx=new Map([["",{unit:"",scale:1}],["angstrom",{unit:"m",scale:1e-10}],["foot",{unit:"m",scale:.3048}],["inch",{unit:"m",scale:.0254}],["mile",{unit:"m",scale:1609.34}],["parsec",{unit:"m",scale:0x6da012f95c9e88}],["yard",{unit:"m",scale:.9144}],["minute",{unit:"s",scale:60}],["hour",{unit:"s",scale:60*60}],["day",{unit:"s",scale:60*60*24}]]);for(let e of["meter","second"])for(let t of su){let{longPrefix:r,prefix:n}=t;if(r===void 0)continue;let i={unit:e[0],scale:10**t.exponent};Qx.set(`${r}${e}`,i),Qx.set(`${n}${e[0]}`,i)}var Es=(e=>(e[e.START=0]="START",e[e.END=1]="END",e))(Es||{});Zx({name:"sharding_indexed",kind:ne.arrayToBytes,resolve(e,t){Ee(e);let r=re(e,"chunk_shape",c=>e1(c,t.chunkShape.length)),n=nr(e,"index_location",c=>kn(c,Es,/^[a-z]+$/),1),i=Array.from(t.chunkShape,(c,l)=>{let u=r[l];if(c%u!==0)throw new Error(`sub-chunk shape of ${JSON.stringify(u)} does not evenly divide outer chunk shape of ${JSON.stringify(t.chunkShape)}`);return c/u}),s=Array.from(i);s.push(2);let o=re(e,"index_codecs",c=>nc(c,{dataType:W.UINT64,chunkShape:s}));if(o.encodedSize[o.encodedSize.length-1]===void 0)throw new Error("index_codecs must specify fixed-size encoding");let a=re(e,"codecs",c=>nc(c,{dataType:t.dataType,chunkShape:r}));return{configuration:{indexCodecs:o,subChunkCodecs:a,subChunkShape:r,subChunkGridShape:i,indexLocation:n},shardingInfo:{subChunkShape:r,subChunkGridShape:i,subChunkCodecs:a}}},getDecodedArrayLayoutInfo(e,t){return e.subChunkCodecs.layoutInfo[0]}});var t1=BigInt("18446744073709551615");function xT(e,t,r){return new qe(e.addRef(),{get:async(n,i)=>{let{indexCodecs:s}=r,o=s.encodedSize[s.encodedSize.length-1],a;switch(r.indexLocation){case Es.START:a={offset:0,length:o};break;case Es.END:a={suffixLength:o};break}let c=await t.read(n,{...i,byteRange:a});if(c===void 0)return{size:0,data:void 0};let l=await tc(r.indexCodecs,new Uint8Array(await c.response.arrayBuffer()),i.signal);return{size:l.byteLength,data:new BigUint64Array(l.buffer,l.byteOffset,l.byteLength/8)}}})}var Qf=class extends be{constructor(t,r,n){super(),this.configuration=t,this.base=n,this.indexCache=this.registerDisposer(xT(r,n,t));let{subChunkGridShape:i}=this.configuration,s=i.length,o=this.configuration.indexCodecs.layoutInfo[0].physicalToLogicalDimension,a=this.indexStrides=new Array(s+1),c=1;for(let l=s;l>=0;--l){let u=o[l];a[u]=c,c*=u===s?2:i[u]}}indexCache;indexStrides;async findKey(t,r){let n=await this.indexCache.get(t.base,r);if(n===void 0)return;let i=this.configuration.subChunkShape.length,{subChunk:s}=t,{indexStrides:o}=this,a=0;for(let u=0;u<i;++u){let f=s[u];a+=f*o[u]}let c=n[a],l=n[a+o[i]];if(!(c===t1&&l===t1))return{offset:Number(c),length:Number(l)}}async stat(t,r){let n=await this.findKey(t,r);if(n!==void 0)return{totalSize:n.length}}async read(t,r){let n=await this.findKey(t,r);if(n!==void 0)return new tt(new ke(this.base,t.base),n).read(r)}getUrl(t){return`subchunk ${JSON.stringify(t.subChunk)} within shard ${this.base.getUrl(t.base)}`}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};pt({name:"sharding_indexed",kind:ne.arrayToBytes,getShardedKvStore(e,t,r){return new Qf(e,t,r)}});pt({name:"transpose",kind:ne.arrayToArray,async decode(e,t,r,n){return r}});var ST=Object.defineProperty,wT=Object.getOwnPropertyDescriptor,ET=(e,t,r,n)=>{for(var i=n>1?void 0:n?wT(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&ST(t,r,i),i};var r1=class extends ye(Te(Ae),rc){chunkKvStore=Wx(this.chunkManager,this.parameters.metadata.codecs,this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url));async download(e,t){e.chunkDataSize=this.spec.chunkDataSize;let{parameters:r}=this,{chunkGridPosition:n}=e,{metadata:i}=r,s="",o=this.spec.rank,{physicalToLogicalDimension:a}=i.codecs.layoutInfo[0],c;i.chunkKeyEncoding===ic.DEFAULT?(s+="c",c=i.dimensionSeparator):(c="",o===0&&(s+="0"));let l=new Array(o),{readChunkShape:u}=i.codecs.layoutInfo[0],{chunkShape:f}=i;for(let d=0;d<o;++d){let m=a[o-1-d];l[m]=Math.floor(n[d]*u[m]/f[m])}for(let d=0;d<o;++d)s+=`${c}${l[d]}`,c=i.dimensionSeparator;let{chunkKvStore:h}=this,p=await h.kvStore.read(h.getChunkKey(n,s),{signal:t});if(p!==void 0){let d=await tc(h.decodeCodecs,new Uint8Array(await p.response.arrayBuffer()),t);await ut(e,t,d)}}};r1=ET([j()],r1);function n1(e){let t=e.match(/^([0-9]+)-([0-9]+)$/);if(t!==null){let r=Number(t[1]),n=Number(t[2]);if(n>=r)return{offset:r,length:n-r}}throw new Error(`Invalid key ${JSON.stringify(e)} for "byte-range:", expected "<begin>-<end>"`)}var sc=class{constructor(t){this.base=t}getUrl(t){return this.base.getUrl()+`|byte-range:${t}`}async stat(t,r){let{length:n}=n1(t);return{totalSize:n}}async read(t,r){let n=n1(t);return new tt(this.base,n).read(r)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}get singleKey(){return!0}};function bT(){return{scheme:"byte-range",description:"byte range slicing",getKvStore(e,t){return Rr(e),{store:new sc(new ke(t.store,t.path)),path:e.suffix??""}}}}Wt.registerKvStoreAdapterProvider(bT);var i1=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),s1=e=>{throw TypeError(e)},IT=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&s1("Object expected");var n,i;r&&(n=t[i1("asyncDispose")]),n===void 0&&(n=t[i1("dispose")],r&&(i=n)),typeof n!="function"&&s1("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},_T=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,l){return l=Error(c),l.name="SuppressedError",l.error=o,l.suppressed=a,l},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};var Si=class{constructor(t,r=`gs://${t}/`,n=Re){this.bucket=t,this.baseUrlForDisplay=r,this.fetchOkImpl=n}getObjectUrl(t){return`https://storage.googleapis.com/storage/v1/b/${this.bucket}/o/${encodeURIComponent(t)}?alt=media&neuroglancer=${vu()}`}stat(t,r){return pn(this,t,this.getObjectUrl(t),r,this.fetchOkImpl)}read(t,r){return ei(this,t,this.getObjectUrl(t),r,this.fetchOkImpl)}async list(t,r){var n=[];try{let{progressListener:o}=r,a=IT(n,o===void 0?void 0:new He(o,{message:`Listing prefix ${this.getUrl(t)}`})),u=await(await this.fetchOkImpl(`https://storage.googleapis.com/storage/v1/b/${this.bucket}/o?delimiter=${encodeURIComponent("/")}&prefix=${encodeURIComponent(t)}&neuroglancerOrigin=${encodeURIComponent(location.origin)}`,{signal:r.signal,progressListener:r.progressListener})).json();Ee(u);let f=nr(u,"prefixes",Yt,[]).map(p=>p.substring(0,p.length-1)),h=nr(u,"items",p=>ct(p,d=>(Ee(d),re(d,"name",et))),[]).filter(p=>!p.endsWith("_$folder$")).map(p=>({key:p}));return{directories:f,entries:h}}catch(o){var i=o,s=!0}finally{_T(n,i,s)}}getUrl(t){return this.baseUrlForDisplay+Ze(t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function CT(e){return{scheme:"gs",description:Ss?"Google Cloud Storage":"Google Cloud Storage (anonymous)",getKvStore(t){let r=(t.suffix??"").match(/^\/\/([^/]+)(\/.*)?$/);if(r===null)throw new Error("Invalid URL, expected `gs://<bucket>/<path>`");let[,n,i]=r;return{store:new Si(n),path:decodeURIComponent((i??"").substring(1))}}}}Wt.registerBaseKvStoreProvider(CT);var oc=class{constructor(t,r,n){this.base=t,this.scheme=r,this.format=n}getUrl(t){return this.validatePath(t),this.base.getUrl()+`|${this.scheme}`}validatePath(t){if(t)throw new Error(`"${this.scheme}:" does not support non-empty path ${JSON.stringify(t)}`)}async stat(t,r){return this.validatePath(t),await this.base.stat(r),{totalSize:void 0}}async read(t,r){return this.validatePath(t),new Gr(this.base,this.format).read(r)}get supportsOffsetReads(){return!1}get supportsSuffixReads(){return!1}get singleKey(){return!0}};async function AT(e){return _0(e.prefix)?[{suffix:"gzip:",description:"gzip-compressed"}]:[]}function o1(e){e.registerFileFormat({prefixLength:3,suffixLength:0,match:AT})}function MT(e,t){return{scheme:e,description:`transparent ${e} decoding`,getKvStore(r,n){return _g(r),{store:new oc(new ke(n.store,n.path),e,t),path:""}}}}Wt.registerKvStoreAdapterProvider(()=>MT("gzip","gzip"));o1(Wt.autoDetectRegistry);xt(Tg,async function(e,t){return{value:await this.get(e.sharedKvStoreContext).kvStoreContext.stat(e.url,t)}});xt(Ng,async function(e,t){let n=await this.get(e.sharedKvStoreContext).kvStoreContext.read(e.url,{...t,byteRange:e.byteRange,throwIfMissing:e.throwIfMissing});if(n===void 0)return{value:void 0};let i=await n.response.arrayBuffer();return{value:{data:i,offset:n.offset,totalSize:n.totalSize},transfers:[i]}});function ac(e,t,r){return e.rpc.promiseInvoke(tu,{sharedKvStoreContext:e.rpcId,url:t},{signal:r.signal,progressListener:r.progressListener})}xt(tu,async function(e,t){let r=this.get(e.sharedKvStoreContext),{store:n,path:i}=r.kvStoreContext.getKvStore(e.url);return{value:await n.list(i,t)}});xt(Dg,async function(e,t){let r=this.get(e.sharedKvStoreContext),{kvStoreContext:n}=r,{url:i}=e,s=bg(i),o;if(s===i){let a=bo(s),c=n.getBaseKvStoreProvider(a);c.completeUrl!==void 0&&(o=await c.completeUrl({url:a,...t}))}else{let a=bo(s),c=n.getKvStoreAdapterProvider(a),l=i.slice(0,i.length-s.length-1),u=n.getKvStore(l);c.completeUrl!==void 0&&(o=await c.completeUrl({url:a,base:u,...t}))}return{value:o}});var wi=class extends ti{list(t,r){return ac(this.sharedKvStoreContext,this.getUrl(t),r)}};fv(wt,wi);function a1(e,t){let{nodes:r}=e,n=we(0,r.length,a=>r[a].path>=t),i=we(Math.min(r.length,n+1),r.length,a=>!r[a].path.startsWith(t)),s={entries:[],directories:[]};for(let a=n;a<i;){let c=r[a],{path:l}=c,u=l.indexOf("/",t.length);if(u===-1)++a;else{u+1===l.length&&s.directories.push(l.slice(0,u));let f=l.substring(0,u+1);a=we(a+1,i,h=>!r[h].path.startsWith(f))}}let o=t.lastIndexOf("/");if("zarr.json".startsWith(t.slice(o+1))){let a=t.substring(0,o+1);if(er(r,a,(l,u)=>St(l,u.path))>=0)s.entries.push({key:a+"zarr.json"});else throw new Error(`Parent node ${JSON.stringify(a)} not found`)}return Eo(s)}var cc,kT={lang:void 0,message:void 0,abortEarly:void 0,abortPipeEarly:void 0};function l1(e){return!e&&!cc?kT:{lang:e?.lang??cc?.lang,message:e?.message,abortEarly:e?.abortEarly??cc?.abortEarly,abortPipeEarly:e?.abortPipeEarly??cc?.abortPipeEarly}}var TT;function NT(e){return TT?.get(e)}var DT;function PT(e){return DT?.get(e)}var RT;function OT(e,t){return RT?.get(e)?.get(t)}function u1(e){let t=typeof e;return t==="string"?`"${e}"`:t==="number"||t==="bigint"||t==="boolean"?`${e}`:t==="object"||t==="function"?(e&&Object.getPrototypeOf(e)?.constructor?.name)??"null":t}function Pe(e,t,r,n,i){let s=i&&"input"in i?i.input:r.value,o=i?.expected??e.expects??null,a=i?.received??u1(s),c={kind:e.kind,type:e.type,input:s,expected:o,received:a,message:`Invalid ${t}: ${o?`Expected ${o} but r`:"R"}eceived ${a}`,requirement:e.requirement,path:i?.path,issues:i?.issues,lang:n.lang,abortEarly:n.abortEarly,abortPipeEarly:n.abortPipeEarly},l=e.kind==="schema",u=i?.message??e.message??OT(e.reference,c.lang)??(l?PT(c.lang):null)??n.message??NT(c.lang);u!==void 0&&(c.message=typeof u=="function"?u(c):u),l&&(r.typed=!1),r.issues?r.issues.push(c):r.issues=[c]}function UT(e,t){return Object.prototype.hasOwnProperty.call(e,t)&&t!=="__proto__"&&t!=="prototype"&&t!=="constructor"}function f1(e,t){let r=[...new Set(e)];return r.length>1?`(${r.join(` ${t} `)})`:r[0]??"never"}function st(e){return e["~standard"]={version:1,vendor:"valibot",validate:t=>e["~run"]({value:t},l1())},e}function LT(e){if(e.path){let t="";for(let r of e.path)if(typeof r.key=="string"||typeof r.key=="number")t?t+=`.${r.key}`:t+=r.key;else return null;return t}return null}var h1=class extends Error{constructor(e){super(e[0].message),this.name="ValiError",this.issues=e}};function p1(e){return e instanceof h1}function eh(e,t){return{kind:"validation",type:"check",reference:eh,async:!1,expects:null,requirement:e,message:t,"~run"(r,n){return r.typed&&!this.requirement(r.value)&&Pe(this,"input",r,n),r}}}function th(e){return{kind:"validation",type:"integer",reference:th,async:!1,expects:null,requirement:Number.isInteger,message:e,"~run"(t,r){return t.typed&&!this.requirement(t.value)&&Pe(this,"integer",t,r),t}}}function bs(e,t){return{kind:"validation",type:"length",reference:bs,async:!1,expects:`${e}`,requirement:e,message:t,"~run"(r,n){return r.typed&&r.value.length!==this.requirement&&Pe(this,"length",r,n,{received:`${r.value.length}`}),r}}}function Ye(e){return{kind:"transformation",type:"transform",reference:Ye,async:!1,operation:e,"~run"(t){return t.value=this.operation(t.value),t}}}function FT(e,t,r){return typeof e.fallback=="function"?e.fallback(t,r):e.fallback}function d1(e){let t={};for(let r of e)if(r.path){let n=LT(r);n?(t.nested||(t.nested={}),Object.prototype.hasOwnProperty.call(t.nested,n)?t.nested[n].push(r.message):t.nested[n]=[r.message]):t.other?t.other.push(r.message):t.other=[r.message]}else t.root?t.root.push(r.message):t.root=[r.message];return t}function m1(e,t,r){return typeof e.default=="function"?e.default(t,r):e.default}function _t(){return st({kind:"schema",type:"any",reference:_t,expects:"any",async:!1,"~run"(e){return e.typed=!0,e}})}function ot(e,t){return st({kind:"schema",type:"array",reference:ot,expects:"Array",async:!1,item:e,message:t,"~run"(r,n){let i=r.value;if(Array.isArray(i)){r.typed=!0,r.value=[];for(let s=0;s<i.length;s++){let o=i[s],a=this.item["~run"]({value:o},n);if(a.issues){let c={type:"array",origin:"value",input:i,key:s,value:o};for(let l of a.issues)l.path?l.path.unshift(c):l.path=[c],r.issues?.push(l);if(r.issues||(r.issues=a.issues),n.abortEarly){r.typed=!1;break}}a.typed||(r.typed=!1),r.value.push(a.value)}}else Pe(this,"type",r,n);return r}})}function rh(e){return st({kind:"schema",type:"bigint",reference:rh,expects:"bigint",async:!1,message:e,"~run"(t,r){return typeof t.value=="bigint"?t.typed=!0:Pe(this,"type",t,r),t}})}function Is(e,t){return st({kind:"schema",type:"instance",reference:Is,expects:e.name,async:!1,class:e,message:t,"~run"(r,n){return r.value instanceof this.class?r.typed=!0:Pe(this,"type",r,n),r}})}function Xt(e,t,r){return st({kind:"schema",type:"map",reference:Xt,expects:"Map",async:!1,key:e,value:t,message:r,"~run"(n,i){let s=n.value;if(s instanceof Map){n.typed=!0,n.value=new Map;for(let[o,a]of s){let c=this.key["~run"]({value:o},i);if(c.issues){let u={type:"map",origin:"key",input:s,key:o,value:a};for(let f of c.issues)f.path?f.path.unshift(u):f.path=[u],n.issues?.push(f);if(n.issues||(n.issues=c.issues),i.abortEarly){n.typed=!1;break}}let l=this.value["~run"]({value:a},i);if(l.issues){let u={type:"map",origin:"value",input:s,key:o,value:a};for(let f of l.issues)f.path?f.path.unshift(u):f.path=[u],n.issues?.push(f);if(n.issues||(n.issues=l.issues),i.abortEarly){n.typed=!1;break}}(!c.typed||!l.typed)&&(n.typed=!1),n.value.set(c.value,l.value)}}else Pe(this,"type",n,i);return n}})}function _s(e,t){return st({kind:"schema",type:"nullable",reference:_s,expects:`(${e.expects} | null)`,async:!1,wrapped:e,default:t,"~run"(r,n){return r.value===null&&(this.default!==void 0&&(r.value=m1(this,r,n)),r.value===null)?(r.typed=!0,r):this.wrapped["~run"](r,n)}})}function nh(e){return st({kind:"schema",type:"number",reference:nh,expects:"number",async:!1,message:e,"~run"(t,r){return typeof t.value=="number"&&!isNaN(t.value)?t.typed=!0:Pe(this,"type",t,r),t}})}function lc(e,t){return st({kind:"schema",type:"picklist",reference:lc,expects:f1(e.map(u1),"|"),async:!1,options:e,message:t,"~run"(r,n){return this.options.includes(r.value)?r.typed=!0:Pe(this,"type",r,n),r}})}function ih(e,t,r){return st({kind:"schema",type:"record",reference:ih,expects:"Object",async:!1,key:e,value:t,message:r,"~run"(n,i){let s=n.value;if(s&&typeof s=="object"){n.typed=!0,n.value={};for(let o in s)if(UT(s,o)){let a=s[o],c=this.key["~run"]({value:o},i);if(c.issues){let u={type:"object",origin:"key",input:s,key:o,value:a};for(let f of c.issues)f.path=[u],n.issues?.push(f);if(n.issues||(n.issues=c.issues),i.abortEarly){n.typed=!1;break}}let l=this.value["~run"]({value:a},i);if(l.issues){let u={type:"object",origin:"value",input:s,key:o,value:a};for(let f of l.issues)f.path?f.path.unshift(u):f.path=[u],n.issues?.push(f);if(n.issues||(n.issues=l.issues),i.abortEarly){n.typed=!1;break}}(!c.typed||!l.typed)&&(n.typed=!1),c.typed&&(n.value[c.value]=l.value)}}else Pe(this,"type",n,i);return n}})}function ur(e,t){return st({kind:"schema",type:"strict_object",reference:ur,expects:"Object",async:!1,entries:e,message:t,"~run"(r,n){let i=r.value;if(i&&typeof i=="object"){r.typed=!0,r.value={};for(let s in this.entries){let o=this.entries[s];if(s in i||(o.type==="exact_optional"||o.type==="optional"||o.type==="nullish")&&o.default!==void 0){let a=s in i?i[s]:m1(o),c=o["~run"]({value:a},n);if(c.issues){let l={type:"object",origin:"value",input:i,key:s,value:a};for(let u of c.issues)u.path?u.path.unshift(l):u.path=[l],r.issues?.push(u);if(r.issues||(r.issues=c.issues),n.abortEarly){r.typed=!1;break}}c.typed||(r.typed=!1),r.value[s]=c.value}else if(o.fallback!==void 0)r.value[s]=FT(o);else if(o.type!=="exact_optional"&&o.type!=="optional"&&o.type!=="nullish"&&(Pe(this,"key",r,n,{input:void 0,expected:`"${s}"`,path:[{type:"object",origin:"key",input:i,key:s,value:i[s]}]}),n.abortEarly))break}if(!r.issues||!n.abortEarly){for(let s in i)if(!Object.prototype.hasOwnProperty.call(this.entries,s)){Pe(this,"key",r,n,{input:s,expected:"never",path:[{type:"object",origin:"key",input:i,key:s,value:i[s]}]});break}}}else Pe(this,"type",r,n);return r}})}function sh(e,t){return st({kind:"schema",type:"strict_tuple",reference:sh,expects:"Array",async:!1,items:e,message:t,"~run"(r,n){let i=r.value;if(Array.isArray(i)){r.typed=!0,r.value=[];for(let s=0;s<this.items.length;s++){let o=i[s],a=this.items[s]["~run"]({value:o},n);if(a.issues){let c={type:"array",origin:"value",input:i,key:s,value:o};for(let l of a.issues)l.path?l.path.unshift(c):l.path=[c],r.issues?.push(l);if(r.issues||(r.issues=a.issues),n.abortEarly){r.typed=!1;break}}a.typed||(r.typed=!1),r.value.push(a.value)}!(r.issues&&n.abortEarly)&&this.items.length<i.length&&Pe(this,"type",r,n,{input:i[this.items.length],expected:"never",path:[{type:"array",origin:"value",input:i,key:this.items.length,value:i[this.items.length]}]})}else Pe(this,"type",r,n);return r}})}function Ue(e){return st({kind:"schema",type:"string",reference:Ue,expects:"string",async:!1,message:e,"~run"(t,r){return typeof t.value=="string"?t.typed=!0:Pe(this,"type",t,r),t}})}function oh(e,t){return st({kind:"schema",type:"tuple",reference:oh,expects:"Array",async:!1,items:e,message:t,"~run"(r,n){let i=r.value;if(Array.isArray(i)){r.typed=!0,r.value=[];for(let s=0;s<this.items.length;s++){let o=i[s],a=this.items[s]["~run"]({value:o},n);if(a.issues){let c={type:"array",origin:"value",input:i,key:s,value:o};for(let l of a.issues)l.path?l.path.unshift(c):l.path=[c],r.issues?.push(l);if(r.issues||(r.issues=a.issues),n.abortEarly){r.typed=!1;break}}a.typed||(r.typed=!1),r.value.push(a.value)}}else Pe(this,"type",r,n);return r}})}function c1(e){let t;if(e)for(let r of e)if(t)for(let n of r.issues)t.push(n);else t=r.issues;return t}function Hr(e,t){return st({kind:"schema",type:"union",reference:Hr,expects:f1(e.map(r=>r.expects),"|"),async:!1,options:e,message:t,"~run"(r,n){let i,s,o;for(let a of this.options){let c=a["~run"]({value:r.value},n);if(c.typed)if(c.issues)s?s.push(c):s=[c];else{i=c;break}else o?o.push(c):o=[c]}if(i)return i;if(s){if(s.length===1)return s[0];Pe(this,"type",r,n,{issues:c1(s)}),r.typed=!0}else{if(o?.length===1)return o[0];Pe(this,"type",r,n,{issues:c1(o)})}return r}})}function g1(e,t,r){let n=e["~run"]({value:t},l1(r));if(n.issues)throw new h1(n.issues);return n.value}function Je(...e){return st({...e[0],pipe:e,"~run"(t,r){for(let n of e)if(n.kind!=="metadata"){if(t.issues&&(n.kind==="schema"||n.kind==="transformation")){t.typed=!1;break}(!t.issues||!r.abortEarly&&!r.abortPipeEarly)&&(t=n["~run"](t,r))}return t}})}var lh;try{lh=new TextDecoder}catch{}var $,zt,A=0;var T1=[],uh=T1,fh=0,ve={},le,Wr,Bt=0,fr=0,Fe,Cr,dt=[],ce,y1={useRecords:!1,mapsAsObjects:!0},Cs=class{},ph=new Cs;ph.name="MessagePack 0xC1";var Xr=!1,v1=2,x1,S1,w1;var Ar=class e{constructor(t){t&&(t.useRecords===!1&&t.mapsAsObjects===void 0&&(t.mapsAsObjects=!0),t.sequential&&t.trusted!==!1&&(t.trusted=!0,!t.structures&&t.useRecords!=!1&&(t.structures=[],t.maxSharedStructures||(t.maxSharedStructures=0))),t.structures?t.structures.sharedLength=t.structures.length:t.getStructures&&((t.structures=[]).uninitialized=!0,t.structures.sharedLength=0),t.int64AsNumber&&(t.int64AsType="number")),Object.assign(this,t)}unpack(t,r){if($)return O1(()=>(fc(),this?this.unpack(t,r):e.prototype.unpack.call(y1,t,r)));!t.buffer&&t.constructor===ArrayBuffer&&(t=typeof Buffer<"u"?Buffer.from(t):new Uint8Array(t)),typeof r=="object"?(zt=r.end||t.length,A=r.start||0):(A=0,zt=r>-1?r:t.length),fh=0,fr=0,Wr=null,uh=T1,Fe=null,$=t;try{ce=t.dataView||(t.dataView=new DataView(t.buffer,t.byteOffset,t.byteLength))}catch(n){throw $=null,t instanceof Uint8Array?n:new Error("Source must be a Uint8Array or Buffer but was a "+(t&&typeof t=="object"?t.constructor.name:typeof t))}if(this instanceof e){if(ve=this,this.structures)return le=this.structures,uc(r);(!le||le.length>0)&&(le=[])}else ve=y1,(!le||le.length>0)&&(le=[]);return uc(r)}unpackMultiple(t,r){let n,i=0;try{Xr=!0;let s=t.length,o=this?this.unpack(t,s):pc.unpack(t,s);if(r){if(r(o,i,A)===!1)return;for(;A<s;)if(i=A,r(uc(),i,A)===!1)return}else{for(n=[o];A<s;)i=A,n.push(uc());return n}}catch(s){throw s.lastPosition=i,s.values=n,s}finally{Xr=!1,fc()}}_mergeStructures(t,r){S1&&(t=S1.call(this,t)),t=t||[],Object.isFrozen(t)&&(t=t.map(n=>n.slice(0)));for(let n=0,i=t.length;n<i;n++){let s=t[n];s&&(s.isShared=!0,n>=32&&(s.highByte=n-32>>5))}t.sharedLength=t.length;for(let n in r||[])if(n>=0){let i=t[n],s=r[n];s&&(i&&((t.restoreStructures||(t.restoreStructures=[]))[n]=i),t[n]=s)}return this.structures=t}decode(t,r){return this.unpack(t,r)}};function uc(e){try{if(!ve.trusted&&!Xr){let r=le.sharedLength||0;r<le.length&&(le.length=r)}let t;if(ve.randomAccessStructure&&$[A]<64&&$[A]>=32&&x1?(t=x1($,A,zt,ve),$=null,!(e&&e.lazy)&&t&&(t=t.toJSON()),A=zt):t=Me(),Fe&&(A=Fe.postBundlePosition,Fe=null),Xr&&(le.restoreStructures=null),A==zt)le&&le.restoreStructures&&E1(),le=null,$=null,Cr&&(Cr=null);else{if(A>zt)throw new Error("Unexpected end of MessagePack data");if(!Xr){let r;try{r=JSON.stringify(t,(n,i)=>typeof i=="bigint"?`${i}n`:i).slice(0,100)}catch(n){r="(JSON view not available "+n+")"}throw new Error("Data read, but end of buffer not reached "+r)}}return t}catch(t){throw le&&le.restoreStructures&&E1(),fc(),(t instanceof RangeError||t.message.startsWith("Unexpected end of buffer")||A>zt)&&(t.incomplete=!0),t}}function E1(){for(let e in le.restoreStructures)le[e]=le.restoreStructures[e];le.restoreStructures=null}function Me(){let e=$[A++];if(e<160)if(e<128){if(e<64)return e;{let t=le[e&63]||ve.getStructures&&N1()[e&63];return t?(t.read||(t.read=dh(t,e&63)),t.read()):e}}else if(e<144)if(e-=128,ve.mapsAsObjects){let t={};for(let r=0;r<e;r++){let n=P1();n==="__proto__"&&(n="__proto_"),t[n]=Me()}return t}else{let t=new Map;for(let r=0;r<e;r++)t.set(Me(),Me());return t}else{e-=144;let t=new Array(e);for(let r=0;r<e;r++)t[r]=Me();return ve.freezeData?Object.freeze(t):t}else if(e<192){let t=e-160;if(fr>=A)return Wr.slice(A-Bt,(A+=t)-Bt);if(fr==0&&zt<140){let r=t<16?mh(t):D1(t);if(r!=null)return r}return hh(t)}else{let t;switch(e){case 192:return null;case 193:return Fe?(t=Me(),t>0?Fe[1].slice(Fe.position1,Fe.position1+=t):Fe[0].slice(Fe.position0,Fe.position0-=t)):ph;case 194:return!1;case 195:return!0;case 196:if(t=$[A++],t===void 0)throw new Error("Unexpected end of buffer");return ch(t);case 197:return t=ce.getUint16(A),A+=2,ch(t);case 198:return t=ce.getUint32(A),A+=4,ch(t);case 199:return vn($[A++]);case 200:return t=ce.getUint16(A),A+=2,vn(t);case 201:return t=ce.getUint32(A),A+=4,vn(t);case 202:if(t=ce.getFloat32(A),ve.useFloat32>2){let r=hc[($[A]&127)<<1|$[A+1]>>7];return A+=4,(r*t+(t>0?.5:-.5)>>0)/r}return A+=4,t;case 203:return t=ce.getFloat64(A),A+=8,t;case 204:return $[A++];case 205:return t=ce.getUint16(A),A+=2,t;case 206:return t=ce.getUint32(A),A+=4,t;case 207:return ve.int64AsType==="number"?(t=ce.getUint32(A)*4294967296,t+=ce.getUint32(A+4)):ve.int64AsType==="string"?t=ce.getBigUint64(A).toString():ve.int64AsType==="auto"?(t=ce.getBigUint64(A),t<=BigInt(2)<<BigInt(52)&&(t=Number(t))):t=ce.getBigUint64(A),A+=8,t;case 208:return ce.getInt8(A++);case 209:return t=ce.getInt16(A),A+=2,t;case 210:return t=ce.getInt32(A),A+=4,t;case 211:return ve.int64AsType==="number"?(t=ce.getInt32(A)*4294967296,t+=ce.getUint32(A+4)):ve.int64AsType==="string"?t=ce.getBigInt64(A).toString():ve.int64AsType==="auto"?(t=ce.getBigInt64(A),t>=BigInt(-2)<<BigInt(52)&&t<=BigInt(2)<<BigInt(52)&&(t=Number(t))):t=ce.getBigInt64(A),A+=8,t;case 212:if(t=$[A++],t==114)return M1($[A++]&63);{let r=dt[t];if(r)return r.read?(A++,r.read(Me())):r.noBuffer?(A++,r()):r($.subarray(A,++A));throw new Error("Unknown extension "+t)}case 213:return t=$[A],t==114?(A++,M1($[A++]&63,$[A++])):vn(2);case 214:return vn(4);case 215:return vn(8);case 216:return vn(16);case 217:return t=$[A++],fr>=A?Wr.slice(A-Bt,(A+=t)-Bt):zT(t);case 218:return t=ce.getUint16(A),A+=2,fr>=A?Wr.slice(A-Bt,(A+=t)-Bt):$T(t);case 219:return t=ce.getUint32(A),A+=4,fr>=A?Wr.slice(A-Bt,(A+=t)-Bt):VT(t);case 220:return t=ce.getUint16(A),A+=2,I1(t);case 221:return t=ce.getUint32(A),A+=4,I1(t);case 222:return t=ce.getUint16(A),A+=2,_1(t);case 223:return t=ce.getUint32(A),A+=4,_1(t);default:if(e>=224)return e-256;if(e===void 0){let r=new Error("Unexpected end of MessagePack data");throw r.incomplete=!0,r}throw new Error("Unknown MessagePack token "+e)}}}var BT=/^[a-zA-Z_$][a-zA-Z\d_$]*$/;function dh(e,t){function r(){if(r.count++>v1){let i;try{i=e.read=new Function("r","return function(){return "+(ve.freezeData?"Object.freeze":"")+"({"+e.map(s=>s==="__proto__"?"__proto_:r()":BT.test(s)?s+":r()":"["+JSON.stringify(s)+"]:r()").join(",")+"})}")(Me)}catch{return v1=1/0,r()}return e.read0=i,e.highByte===0&&(e.read=b1(t,e.read)),i()}let n={};for(let i=0,s=e.length;i<s;i++){let o=e[i];o==="__proto__"&&(o="__proto_"),n[o]=Me()}return ve.freezeData?Object.freeze(n):n}return r.count=0,e.read0=r,e.highByte===0?b1(t,r):r}var b1=(e,t)=>function(){let r=$[A++];if(r===0)return t();let n=e<32?-(e+(r<<5)):e+(r<<5),i=le[n]||N1()[n];if(!i)throw new Error("Record id is not defined for "+n);return i.read||(i.read=dh(i,e)),i.read()};function N1(){let e=O1(()=>($=null,ve.getStructures()));return le=ve._mergeStructures(e,le)}var hh=As,zT=As,$T=As,VT=As;function As(e){let t;if(e<16&&(t=mh(e)))return t;if(e>64&&lh)return lh.decode($.subarray(A,A+=e));let r=A+e,n=[];for(t="";A<r;){let i=$[A++];if((i&128)===0)n.push(i);else if((i&224)===192){let s=$[A++]&63,o=(i&31)<<6|s;o<128?n.push(65533):n.push(o)}else if((i&240)===224){let s=$[A++]&63,o=$[A++]&63,a=(i&31)<<12|s<<6|o;a<2048||a>=55296&&a<=57343?n.push(65533):n.push(a)}else if((i&248)===240){let s=$[A++]&63,o=$[A++]&63,a=$[A++]&63,c=(i&7)<<18|s<<12|o<<6|a;c<65536||c>1114111?n.push(65533):(c>65535&&(c-=65536,n.push(c>>>10&1023|55296),c=56320|c&1023),n.push(c))}else n.push(65533);n.length>=4096&&(t+=Le.apply(String,n),n.length=0)}return n.length>0&&(t+=Le.apply(String,n)),t}function I1(e){let t=new Array(e);for(let r=0;r<e;r++)t[r]=Me();return ve.freezeData?Object.freeze(t):t}function _1(e){if(ve.mapsAsObjects){let t={};for(let r=0;r<e;r++){let n=P1();n==="__proto__"&&(n="__proto_"),t[n]=Me()}return t}else{let t=new Map;for(let r=0;r<e;r++)t.set(Me(),Me());return t}}var Le=String.fromCharCode;function D1(e){let t=A,r=new Array(e);for(let n=0;n<e;n++){let i=$[A++];if((i&128)>0){A=t;return}r[n]=i}return Le.apply(String,r)}function mh(e){if(e<4)if(e<2){if(e===0)return"";{let t=$[A++];if((t&128)>1){A-=1;return}return Le(t)}}else{let t=$[A++],r=$[A++];if((t&128)>0||(r&128)>0){A-=2;return}if(e<3)return Le(t,r);let n=$[A++];if((n&128)>0){A-=3;return}return Le(t,r,n)}else{let t=$[A++],r=$[A++],n=$[A++],i=$[A++];if((t&128)>0||(r&128)>0||(n&128)>0||(i&128)>0){A-=4;return}if(e<6){if(e===4)return Le(t,r,n,i);{let s=$[A++];if((s&128)>0){A-=5;return}return Le(t,r,n,i,s)}}else if(e<8){let s=$[A++],o=$[A++];if((s&128)>0||(o&128)>0){A-=6;return}if(e<7)return Le(t,r,n,i,s,o);let a=$[A++];if((a&128)>0){A-=7;return}return Le(t,r,n,i,s,o,a)}else{let s=$[A++],o=$[A++],a=$[A++],c=$[A++];if((s&128)>0||(o&128)>0||(a&128)>0||(c&128)>0){A-=8;return}if(e<10){if(e===8)return Le(t,r,n,i,s,o,a,c);{let l=$[A++];if((l&128)>0){A-=9;return}return Le(t,r,n,i,s,o,a,c,l)}}else if(e<12){let l=$[A++],u=$[A++];if((l&128)>0||(u&128)>0){A-=10;return}if(e<11)return Le(t,r,n,i,s,o,a,c,l,u);let f=$[A++];if((f&128)>0){A-=11;return}return Le(t,r,n,i,s,o,a,c,l,u,f)}else{let l=$[A++],u=$[A++],f=$[A++],h=$[A++];if((l&128)>0||(u&128)>0||(f&128)>0||(h&128)>0){A-=12;return}if(e<14){if(e===12)return Le(t,r,n,i,s,o,a,c,l,u,f,h);{let p=$[A++];if((p&128)>0){A-=13;return}return Le(t,r,n,i,s,o,a,c,l,u,f,h,p)}}else{let p=$[A++],d=$[A++];if((p&128)>0||(d&128)>0){A-=14;return}if(e<15)return Le(t,r,n,i,s,o,a,c,l,u,f,h,p,d);let m=$[A++];if((m&128)>0){A-=15;return}return Le(t,r,n,i,s,o,a,c,l,u,f,h,p,d,m)}}}}}function C1(){let e=$[A++],t;if(e<192)t=e-160;else switch(e){case 217:t=$[A++];break;case 218:t=ce.getUint16(A),A+=2;break;case 219:t=ce.getUint32(A),A+=4;break;default:throw new Error("Expected string")}return As(t)}function ch(e){return ve.copyBuffers?Uint8Array.prototype.slice.call($,A,A+=e):$.subarray(A,A+=e)}function vn(e){let t=$[A++];if(dt[t]){let r;return dt[t]($.subarray(A,r=A+=e),n=>{A=n;try{return Me()}finally{A=r}})}else throw new Error("Unknown extension type "+t)}var A1=new Array(4096);function P1(){let e=$[A++];if(e>=160&&e<192){if(e=e-160,fr>=A)return Wr.slice(A-Bt,(A+=e)-Bt);if(!(fr==0&&zt<180))return hh(e)}else return A--,R1(Me());let t=(e<<5^(e>1?ce.getUint16(A):e>0?$[A]:0))&4095,r=A1[t],n=A,i=A+e-3,s,o=0;if(r&&r.bytes==e){for(;n<i;){if(s=ce.getUint32(n),s!=r[o++]){n=1879048192;break}n+=4}for(i+=3;n<i;)if(s=$[n++],s!=r[o++]){n=1879048192;break}if(n===i)return A=n,r.string;i-=3,n=A}for(r=[],A1[t]=r,r.bytes=e;n<i;)s=ce.getUint32(n),r.push(s),n+=4;for(i+=3;n<i;)s=$[n++],r.push(s);let a=e<16?mh(e):D1(e);return a!=null?r.string=a:r.string=hh(e)}function R1(e){if(typeof e=="string")return e;if(typeof e=="number"||typeof e=="boolean"||typeof e=="bigint")return e.toString();if(e==null)return e+"";if(ve.allowArraysInMapKeys&&Array.isArray(e)&&e.flat().every(t=>["string","number","boolean","bigint"].includes(typeof t)))return e.flat().toString();throw new Error(`Invalid property type for record: ${typeof e}`)}var M1=(e,t)=>{let r=Me().map(R1),n=e;t!==void 0&&(e=e<32?-((t<<5)+e):(t<<5)+e,r.highByte=t);let i=le[e];return i&&(i.isShared||Xr)&&((le.restoreStructures||(le.restoreStructures=[]))[e]=i),le[e]=r,r.read=dh(r,n),(r.read0||r.read)()};dt[0]=()=>{};dt[0].noBuffer=!0;dt[66]=e=>{let t=e.byteLength%8||8,r=BigInt(e[0]&128?e[0]-256:e[0]);for(let n=1;n<t;n++)r<<=BigInt(8),r+=BigInt(e[n]);if(e.byteLength!==t){let n=new DataView(e.buffer,e.byteOffset,e.byteLength),i=(s,o)=>{let a=o-s;if(a<=40){let f=n.getBigUint64(s);for(let h=s+8;h<o;h+=8)f<<=BigInt(64),f|=n.getBigUint64(h);return f}let c=s+(a>>4<<3),l=i(s,c),u=i(c,o);return l<<BigInt((o-c)*8)|u};r=r<<BigInt((n.byteLength-t)*8)|i(t,n.byteLength)}return r};var k1={Error,EvalError,RangeError,ReferenceError,SyntaxError,TypeError,URIError,AggregateError:typeof AggregateError=="function"?AggregateError:null};dt[101]=()=>{let e=Me();if(!k1[e[0]]){let t=Error(e[1],{cause:e[2]});return t.name=e[0],t}return k1[e[0]](e[1],{cause:e[2]})};dt[105]=e=>{if(ve.structuredClone===!1)throw new Error("Structured clone extension is disabled");let t=ce.getUint32(A-4);Cr||(Cr=new Map);let r=$[A],n;r>=144&&r<160||r==220||r==221?n=[]:r>=128&&r<144||r==222||r==223?n=new Map:(r>=199&&r<=201||r>=212&&r<=216)&&$[A+1]===115?n=new Set:n={};let i={target:n};Cr.set(t,i);let s=Me();if(i.used)Object.assign(n,s);else return i.target=s;if(n instanceof Map)for(let[o,a]of s.entries())n.set(o,a);if(n instanceof Set)for(let o of Array.from(s))n.add(o);return n};dt[112]=e=>{if(ve.structuredClone===!1)throw new Error("Structured clone extension is disabled");let t=ce.getUint32(A-4),r=Cr.get(t);return r.used=!0,r.target};dt[115]=()=>new Set(Me());var gh=["Int8","Uint8","Uint8Clamped","Int16","Uint16","Int32","Uint32","Float32","Float64","BigInt64","BigUint64"].map(e=>e+"Array"),jT=typeof globalThis=="object"?globalThis:window;dt[116]=e=>{let t=e[0],r=Uint8Array.prototype.slice.call(e,1).buffer,n=gh[t];if(!n){if(t===16)return r;if(t===17)return new DataView(r);throw new Error("Could not find typed array for code "+t)}return new jT[n](r)};dt[120]=()=>{let e=Me();return new RegExp(e[0],e[1])};var GT=[];dt[98]=e=>{let t=(e[0]<<24)+(e[1]<<16)+(e[2]<<8)+e[3],r=A;return A+=t-e.length,Fe=GT,Fe=[C1(),C1()],Fe.position0=0,Fe.position1=0,Fe.postBundlePosition=A,A=r,Me()};dt[255]=e=>e.length==4?new Date((e[0]*16777216+(e[1]<<16)+(e[2]<<8)+e[3])*1e3):e.length==8?new Date(((e[0]<<22)+(e[1]<<14)+(e[2]<<6)+(e[3]>>2))/1e6+((e[3]&3)*4294967296+e[4]*16777216+(e[5]<<16)+(e[6]<<8)+e[7])*1e3):e.length==12?new Date(((e[0]<<24)+(e[1]<<16)+(e[2]<<8)+e[3])/1e6+((e[4]&128?-281474976710656:0)+e[6]*1099511627776+e[7]*4294967296+e[8]*16777216+(e[9]<<16)+(e[10]<<8)+e[11])*1e3):new Date("invalid");function O1(e){w1&&w1();let t=zt,r=A,n=fh,i=Bt,s=fr,o=Wr,a=uh,c=Cr,l=Fe,u=new Uint8Array($.slice(0,zt)),f=le,h=le.slice(0,le.length),p=ve,d=Xr,m=e();return zt=t,A=r,fh=n,Bt=i,fr=s,Wr=o,uh=a,Cr=c,Fe=l,$=u,Xr=d,le=f,le.splice(0,le.length,...h),ve=p,ce=new DataView($.buffer,$.byteOffset,$.byteLength),m}function fc(){$=null,Cr=null,le=null}var hc=new Array(147);for(let e=0;e<256;e++)hc[e]=+("1e"+Math.floor(45.15-e*.30103));var pc=new Ar({useRecords:!1}),qT=pc.unpack,KT=pc.unpackMultiple,YT=pc.unpack,dc={NEVER:0,ALWAYS:1,DECIMAL_ROUND:3,DECIMAL_FIT:4},JT=new Float32Array(1),qG=new Uint8Array(JT.buffer,0,4);var gc;try{gc=new TextEncoder}catch{}var xh,Sh,bi=typeof Buffer<"u",mc=bi?function(e){return Buffer.allocUnsafeSlow(e)}:Uint8Array,B1=bi?Buffer:Uint8Array,U1=bi?4294967296:2144337920,D,Ms,xe,T=0,Qe,_e=null,HT,WT=21760,XT=/[\u0080-\uFFFF]/,Ei=Symbol("record-id"),ks=class extends Ar{constructor(t){super(t),this.offset=0;let r,n,i,s,o,a=B1.prototype.utf8Write?function(S,O){return D.utf8Write(S,O,D.byteLength-O)}:gc&&gc.encodeInto?function(S,O){return gc.encodeInto(S,D.subarray(O)).written}:!1,c=this;t||(t={});let l=t&&t.sequential,u=t.structures||t.saveStructures,f=t.maxSharedStructures;if(f==null&&(f=u?32:0),f>8160)throw new Error("Maximum maxSharedStructure is 8160");t.structuredClone&&t.moreTypes==null&&(this.moreTypes=!0);let h=t.maxOwnStructures;h==null&&(h=u?32:64),!this.structures&&t.useRecords!=!1&&(this.structures=[]);let p=f>32||h+f>64,d=f+64,m=f+h+64;if(m>8256)throw new Error("Maximum maxSharedStructure + maxOwnStructure is 8192");let g=[],y=0,I=0;this.pack=this.encode=function(S,O){if(D||(D=new mc(8192),xe=D.dataView||(D.dataView=new DataView(D.buffer,0,8192)),T=0),Qe=D.length-10,Qe-T<2048?(D=new mc(D.length),xe=D.dataView||(D.dataView=new DataView(D.buffer,0,D.length)),Qe=D.length-10,T=0):T=T+7&2147483640,n=T,O&j1&&(T+=O&255),o=c.structuredClone?new Map:null,c.bundleStrings&&typeof S!="string"?(_e=[],_e.size=1/0):_e=null,s=c.structures,s){s.uninitialized&&(s=c._mergeStructures(c.getStructures()));let N=s.sharedLength||0;if(N>f)throw new Error("Shared structures is larger than maximum shared structures, try increasing maxSharedStructures to "+s.sharedLength);if(!s.transitions){s.transitions=Object.create(null);for(let U=0;U<N;U++){let L=s[U];if(!L)continue;let G,V=s.transitions;for(let Y=0,J=L.length;Y<J;Y++){let de=L[Y];G=V[de],G||(G=V[de]=Object.create(null)),V=G}V[Ei]=U+64}this.lastNamedStructuresLength=N}l||(s.nextId=N+64)}i&&(i=!1);let R;try{c.randomAccessStructure&&!c.readOnlyStructures&&S&&typeof S=="object"?S.constructor===Object?F(S):S.constructor!==Map&&!Array.isArray(S)&&!Sh.some(U=>S instanceof U)?F(S.toJSON?S.toJSON():S):b(S):b(S);let N=_e;if(_e&&F1(n,b,0),o&&o.idsToInsert){let U=o.idsToInsert.sort((Y,J)=>Y.offset>J.offset?1:-1),L=U.length,G=-1;for(;N&&L>0;){let Y=U[--L].offset+n;Y<N.stringsPosition+n&&G===-1&&(G=0),Y>N.position+n?G>=0&&(G+=6):(G>=0&&(xe.setUint32(N.position+n,xe.getUint32(N.position+n)+G),G=-1),N=N.previous,L++)}G>=0&&N&&xe.setUint32(N.position+n,xe.getUint32(N.position+n)+G),T+=U.length*6,T>Qe&&k(T),c.offset=T;let V=ZT(D.subarray(n,T),U);return o=null,V}return c.offset=T,O&$1?(D.start=n,D.end=T,D):D.subarray(n,T)}catch(N){throw R=N,N}finally{if(s&&(_(),i&&c.saveStructures)){let N=s.sharedLength||0,U=D.subarray(n,T),L=QT(s,c);if(!R)return c.saveStructures(L,L.isCompatible)===!1?(s.uninitialized=!0,c.pack(S,O)):(c.lastNamedStructuresLength=N,D.length>1073741824&&(D=null),U)}D.length>1073741824&&(D=null),O&V1&&(T=n)}};let _=()=>{I<10&&I++;let S=s.sharedLength||0;if(s.length>S&&!l&&(s.length=S),y>1e4)s.transitions=null,I=0,y=0,g.length>0&&(g=[]);else if(g.length>0&&!l){for(let O=0,R=g.length;O<R;O++)g[O][Ei]=0;g=[]}},E=S=>{var O=S.length;O<16?D[T++]=144|O:O<65536?(D[T++]=220,D[T++]=O>>8,D[T++]=O&255):(D[T++]=221,xe.setUint32(T,O),T+=4);for(let R=0;R<O;R++)b(S[R])},b=S=>{T>Qe&&(D=k(T));var O=typeof S,R;if(O==="string"){let N=S.length;if(_e&&N>=4&&N<4096){if((_e.size+=N)>WT){let V,Y=(_e[0]?_e[0].length*3+_e[1].length:0)+10;T+Y>Qe&&(D=k(T+Y));let J;_e.position?(J=_e,D[T]=200,T+=3,D[T++]=98,V=T-n,T+=4,F1(n,b,0),xe.setUint16(V+n-3,T-n-V)):(D[T++]=214,D[T++]=98,V=T-n,T+=4),_e=["",""],_e.previous=J,_e.size=0,_e.position=V}let G=XT.test(S);_e[G?0:1]+=S,D[T++]=193,b(G?-N:N);return}let U;N<32?U=1:N<256?U=2:N<65536?U=3:U=5;let L=N*3;if(T+L>Qe&&(D=k(T+L)),N<64||!a){let G,V,Y,J=T+U;for(G=0;G<N;G++)V=S.charCodeAt(G),V<128?D[J++]=V:V<2048?(D[J++]=V>>6|192,D[J++]=V&63|128):(V&64512)===55296&&((Y=S.charCodeAt(G+1))&64512)===56320?(V=65536+((V&1023)<<10)+(Y&1023),G++,D[J++]=V>>18|240,D[J++]=V>>12&63|128,D[J++]=V>>6&63|128,D[J++]=V&63|128):(D[J++]=V>>12|224,D[J++]=V>>6&63|128,D[J++]=V&63|128);R=J-T-U}else R=a(S,T+U);R<32?D[T++]=160|R:R<256?(U<2&&D.copyWithin(T+2,T+1,T+1+R),D[T++]=217,D[T++]=R):R<65536?(U<3&&D.copyWithin(T+3,T+2,T+2+R),D[T++]=218,D[T++]=R>>8,D[T++]=R&255):(U<5&&D.copyWithin(T+5,T+3,T+3+R),D[T++]=219,xe.setUint32(T,R),T+=4),T+=R}else if(O==="number")if(S>>>0===S)S<32||S<128&&this.useRecords===!1||S<64&&!this.randomAccessStructure?D[T++]=S:S<256?(D[T++]=204,D[T++]=S):S<65536?(D[T++]=205,D[T++]=S>>8,D[T++]=S&255):(D[T++]=206,xe.setUint32(T,S),T+=4);else if(S>>0===S)S>=-32?D[T++]=256+S:S>=-128?(D[T++]=208,D[T++]=S+256):S>=-32768?(D[T++]=209,xe.setInt16(T,S),T+=2):(D[T++]=210,xe.setInt32(T,S),T+=4);else{let N;if((N=this.useFloat32)>0&&S<4294967296&&S>=-2147483648){D[T++]=202,xe.setFloat32(T,S);let U;if(N<4||(U=S*hc[(D[T]&127)<<1|D[T+1]>>7])>>0===U){T+=4;return}else T--}D[T++]=203,xe.setFloat64(T,S),T+=8}else if(O==="object"||O==="function")if(!S)D[T++]=192;else{if(o){let U=o.get(S);if(U){if(!U.id){let L=o.idsToInsert||(o.idsToInsert=[]);U.id=L.push(U)}D[T++]=214,D[T++]=112,xe.setUint32(T,U.id),T+=4;return}else o.set(S,{offset:T-n})}let N=S.constructor;if(N===Object)x(S);else if(N===Array)E(S);else if(N===Map)if(this.mapAsEmptyObject)D[T++]=128;else{R=S.size,R<16?D[T++]=128|R:R<65536?(D[T++]=222,D[T++]=R>>8,D[T++]=R&255):(D[T++]=223,xe.setUint32(T,R),T+=4);for(let[U,L]of S)b(U),b(L)}else{for(let U=0,L=xh.length;U<L;U++){let G=Sh[U];if(S instanceof G){let V=xh[U];if(V.write){V.type&&(D[T++]=212,D[T++]=V.type,D[T++]=0);let Ce=V.write.call(this,S);Ce===S?Array.isArray(S)?E(S):x(S):b(Ce);return}let Y=D,J=xe,de=T;D=null;let ie;try{ie=V.pack.call(this,S,Ce=>(D=Y,Y=null,T+=Ce,T>Qe&&k(T),{target:D,targetView:xe,position:T-Ce}),b)}finally{Y&&(D=Y,xe=J,T=de,Qe=D.length-10)}ie&&(ie.length+T>Qe&&k(ie.length+T),T=L1(ie,D,T,V.type));return}}if(Array.isArray(S))E(S);else{if(S.toJSON){let U=S.toJSON();if(U!==S)return b(U)}if(O==="function")return b(this.writeFunction&&this.writeFunction(S));x(S)}}}else if(O==="boolean")D[T++]=S?195:194;else if(O==="bigint"){if(S<9223372036854776e3&&S>=-9223372036854776e3)D[T++]=211,xe.setBigInt64(T,S);else if(S<18446744073709552e3&&S>0)D[T++]=207,xe.setBigUint64(T,S);else if(this.largeBigIntToFloat)D[T++]=203,xe.setFloat64(T,Number(S));else{if(this.largeBigIntToString)return b(S.toString());if(this.useBigIntExtension||this.moreTypes){let N=S<0?BigInt(-1):BigInt(0),U;if(S>>BigInt(65536)===N){let L=BigInt(18446744073709552e3)-BigInt(1),G=[];for(;G.push(S&L),S>>BigInt(63)!==N;)S>>=BigInt(64);U=new Uint8Array(new BigUint64Array(G).buffer),U.reverse()}else{let L=S<0,G=(L?~S:S).toString(16);if(G.length%2?G="0"+G:parseInt(G.charAt(0),16)>=8&&(G="00"+G),bi)U=Buffer.from(G,"hex");else{U=new Uint8Array(G.length/2);for(let V=0;V<U.length;V++)U[V]=parseInt(G.slice(V*2,V*2+2),16)}if(L)for(let V=0;V<U.length;V++)U[V]=~U[V]}U.length+T>Qe&&k(U.length+T),T=L1(U,D,T,66);return}else throw new RangeError(S+" was too large to fit in MessagePack 64-bit integer format, use useBigIntExtension, or set largeBigIntToFloat to convert to float-64, or set largeBigIntToString to convert to string")}T+=8}else if(O==="undefined")this.encodeUndefinedAsNil?D[T++]=192:(D[T++]=212,D[T++]=0,D[T++]=0);else throw new Error("Unknown type: "+O)},C=this.variableMapSize||this.coercibleKeyAsNumber||this.skipValues?S=>{let O;if(this.skipValues){O=[];for(let U in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(U))&&!this.skipValues.includes(S[U])&&O.push(U)}else O=Object.keys(S);let R=O.length;R<16?D[T++]=128|R:R<65536?(D[T++]=222,D[T++]=R>>8,D[T++]=R&255):(D[T++]=223,xe.setUint32(T,R),T+=4);let N;if(this.coercibleKeyAsNumber)for(let U=0;U<R;U++){N=O[U];let L=Number(N);b(isNaN(L)?N:L),b(S[N])}else for(let U=0;U<R;U++)b(N=O[U]),b(S[N])}:S=>{D[T++]=222;let O=T-n;T+=2;let R=0;for(let N in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(N))&&(b(N),b(S[N]),R++);if(R>65535)throw new Error('Object is too large to serialize with fast 16-bit map size, use the "variableMapSize" option to serialize this object');D[O+++n]=R>>8,D[O+n]=R&255},v=this.useRecords===!1?C:t.progressiveRecords&&!p?S=>{let O,R=s.transitions||(s.transitions=Object.create(null)),N=T++-n,U;for(let L in S)if(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(L)){if(O=R[L],O)R=O;else{let G=Object.keys(S),V=R;R=s.transitions;let Y=0;for(let J=0,de=G.length;J<de;J++){let ie=G[J];O=R[ie],O||(O=R[ie]=Object.create(null),Y++),R=O}N+n+1==T?(T--,M(R,G,Y)):P(R,G,N,Y),U=!0,R=V[L]}b(S[L])}if(!U){let L=R[Ei];L?D[N+n]=L:P(R,Object.keys(S),N,0)}}:S=>{let O,R=s.transitions||(s.transitions=Object.create(null)),N=0;for(let L in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(L))&&(O=R[L],O||(O=R[L]=Object.create(null),N++),R=O);let U=R[Ei];U?U>=96&&p?(D[T++]=((U-=96)&31)+96,D[T++]=U>>5):D[T++]=U:M(R,R.__keys__||Object.keys(S),N);for(let L in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(L))&&b(S[L])},w=typeof this.useRecords=="function"&&this.useRecords,x=w?S=>{w(S)?v(S):C(S)}:v,k=S=>{let O;if(S>16777216){if(S-n>U1)throw new Error("Packed buffer would be larger than maximum buffer size");O=Math.min(U1,Math.round(Math.max((S-n)*(S>67108864?1.25:2),4194304)/4096)*4096)}else O=(Math.max(S-n<<2,D.length-1)>>12)+1<<12;let R=new mc(O);return xe=R.dataView||(R.dataView=new DataView(R.buffer,0,O)),S=Math.min(S,D.length),D.copy?D.copy(R,0,n,S):R.set(D.slice(n,S)),T-=n,n=0,Qe=R.length-10,D=R},M=(S,O,R)=>{let N=s.nextId;N||(N=64),N<d&&this.shouldShareStructure&&!this.shouldShareStructure(O)?(N=s.nextOwnId,N<m||(N=d),s.nextOwnId=N+1):(N>=m&&(N=d),s.nextId=N+1);let U=O.highByte=N>=96&&p?N-96>>5:-1;S[Ei]=N,S.__keys__=O,s[N-64]=O,N<d?(O.isShared=!0,s.sharedLength=N-63,i=!0,U>=0?(D[T++]=(N&31)+96,D[T++]=U):D[T++]=N):(U>=0?(D[T++]=213,D[T++]=114,D[T++]=(N&31)+96,D[T++]=U):(D[T++]=212,D[T++]=114,D[T++]=N),R&&(y+=I*R),g.length>=h&&(g.shift()[Ei]=0),g.push(S),b(O))},P=(S,O,R,N)=>{let U=D,L=T,G=Qe,V=n;D=Ms,T=0,n=0,D||(Ms=D=new mc(8192)),Qe=D.length-10,M(S,O,N),Ms=D;let Y=T;if(D=U,T=L,Qe=G,n=V,Y>1){let J=T+Y-1;J>Qe&&k(J);let de=R+n;D.copyWithin(de+Y,de+1,T),D.set(Ms.slice(0,Y),de),T=J}else D[R+n]=Ms[0]},F=S=>{let O=HT(S,D,n,T,s,k,(R,N,U)=>{if(U)return i=!0;T=N;let L=D;return b(R),_(),L!==D?{position:T,targetView:xe,target:D}:T},this);if(O===0)return x(S);T=O}}useBuffer(t){D=t,D.dataView||(D.dataView=new DataView(D.buffer,D.byteOffset,D.byteLength)),xe=D.dataView,T=0}set position(t){T=t}get position(){return T}clearSharedData(){this.structures&&(this.structures=[]),this.typedStructs&&(this.typedStructs=[])}};Sh=[Date,Set,Error,RegExp,ArrayBuffer,Object.getPrototypeOf(Uint8Array.prototype).constructor,DataView,Cs];xh=[{pack(e,t,r){let n=e.getTime()/1e3;if((this.useTimestamp32||e.getMilliseconds()===0)&&n>=0&&n<4294967296){let{target:i,targetView:s,position:o}=t(6);i[o++]=214,i[o++]=255,s.setUint32(o,n)}else if(n>0&&n<4294967296){let{target:i,targetView:s,position:o}=t(10);i[o++]=215,i[o++]=255,s.setUint32(o,e.getMilliseconds()*4e6+(n/1e3/4294967296>>0)),s.setUint32(o+4,n)}else if(isNaN(n)){if(this.onInvalidDate)return t(0),r(this.onInvalidDate());let{target:i,targetView:s,position:o}=t(3);i[o++]=212,i[o++]=255,i[o++]=255}else{let{target:i,targetView:s,position:o}=t(15);i[o++]=199,i[o++]=12,i[o++]=255,s.setUint32(o,e.getMilliseconds()*1e6),s.setBigInt64(o+4,BigInt(Math.floor(n)))}}},{pack(e,t,r){if(this.setAsEmptyObject)return t(0),r({});let n=Array.from(e),{target:i,position:s}=t(this.moreTypes?3:0);this.moreTypes&&(i[s++]=212,i[s++]=115,i[s++]=0),r(n)}},{pack(e,t,r){let{target:n,position:i}=t(this.moreTypes?3:0);this.moreTypes&&(n[i++]=212,n[i++]=101,n[i++]=0),r([e.name,e.message,e.cause])}},{pack(e,t,r){let{target:n,position:i}=t(this.moreTypes?3:0);this.moreTypes&&(n[i++]=212,n[i++]=120,n[i++]=0),r([e.source,e.flags])}},{pack(e,t){this.moreTypes?yh(e,16,t):vh(bi?Buffer.from(e):new Uint8Array(e),t)}},{pack(e,t){let r=e.constructor;r!==B1&&this.moreTypes?yh(e,gh.indexOf(r.name),t):vh(e,t)}},{pack(e,t){this.moreTypes?yh(e,17,t):vh(bi?Buffer.from(e):new Uint8Array(e),t)}},{pack(e,t){let{target:r,position:n}=t(1);r[n]=193}}];function yh(e,t,r,n){let i=e.byteLength;if(i+1<256){var{target:s,position:o}=r(4+i);s[o++]=199,s[o++]=i+1}else if(i+1<65536){var{target:s,position:o}=r(5+i);s[o++]=200,s[o++]=i+1>>8,s[o++]=i+1&255}else{var{target:s,position:o,targetView:a}=r(7+i);s[o++]=201,a.setUint32(o,i+1),o+=4}s[o++]=116,s[o++]=t,e.buffer||(e=new Uint8Array(e)),s.set(new Uint8Array(e.buffer,e.byteOffset,e.byteLength),o)}function vh(e,t){let r=e.byteLength;var n,i;if(r<256){var{target:n,position:i}=t(r+2);n[i++]=196,n[i++]=r}else if(r<65536){var{target:n,position:i}=t(r+3);n[i++]=197,n[i++]=r>>8,n[i++]=r&255}else{var{target:n,position:i,targetView:s}=t(r+5);n[i++]=198,s.setUint32(i,r),i+=4}n.set(e,i)}function L1(e,t,r,n){let i=e.length;switch(i){case 1:t[r++]=212;break;case 2:t[r++]=213;break;case 4:t[r++]=214;break;case 8:t[r++]=215;break;case 16:t[r++]=216;break;default:i<256?(t[r++]=199,t[r++]=i):i<65536?(t[r++]=200,t[r++]=i>>8,t[r++]=i&255):(t[r++]=201,t[r++]=i>>24,t[r++]=i>>16&255,t[r++]=i>>8&255,t[r++]=i&255)}return t[r++]=n,t.set(e,r),r+=i,r}function ZT(e,t){let r,n=t.length*6,i=e.length-n;for(;r=t.pop();){let s=r.offset,o=r.id;e.copyWithin(s+n,s,i),n-=6;let a=s+n;e[a++]=214,e[a++]=105,e[a++]=o>>24,e[a++]=o>>16&255,e[a++]=o>>8&255,e[a++]=o&255,i=s}return e}function F1(e,t,r){if(_e.length>0){xe.setUint32(_e.position+e,T+r-_e.position-e),_e.stringsPosition=T-e;let n=_e;_e=null,t(n[0]),t(n[1])}}function QT(e,t){return e.isCompatible=r=>{let n=!r||(t.lastNamedStructuresLength||0)===r.length;return n||t._mergeStructures(r),n},e}var z1=new ks({useRecords:!1}),eN=z1.pack,tN=z1.pack;var{NEVER:rN,ALWAYS:nN,DECIMAL_ROUND:iN,DECIMAL_FIT:sN}=dc,$1=512,V1=1024,j1=2048;var G1="0123456789ABCDEFGHJKMNPQRSTVWXYZ";function wh(e){let t=e.length,r=0,n="",i=0;for(let s=0;s<t;++s)for(r=r<<8|e[s],i+=8;i>=5;i-=5)n+=G1[r>>>i-5&31];return i>0&&(n+=G1[r<<5-i&31]),n}var yc=1,Ts=Uint8Array.of(73,67,69,240,159,167,138,67,72,85,78,75),K1=24,q1=Ts.length+K1+1+1+1;async function oN(e,t,r,n){if(e.byteLength<q1)throw new Error(`Expected icechunk header of ${q1} bytes, but received: ${e.byteLength} bytes`);let i=new DataView(e),s=0;for(let u=0,f=Ts.length;u<f;++u)if(i.getUint8(u)!==Ts[u])throw new Error(`Expected magic bytes of ${Ts.join()} but received: ${new Uint8Array(e,0,f).join()}`);s+=Ts.length,s+=K1;let o=i.getUint8(s++);if(o>t)throw new Error(`Expected version <= ${t} but received: ${o}`);let a=i.getUint8(s++);if(a!==r)throw new Error(`Expected file type of ${r}, but received: ${a}`);let c=i.getUint8(s++),l=new Uint8Array(e,s);switch(c){case 0:break;case 1:l=await ue(qr,n,[e],l),l=new Uint8Array(l.buffer,l.byteOffset,l.byteLength);break;default:throw new Error(`Unknown compression method: ${c}`)}return{content:l,specVersion:o}}async function vc(e,t,r,n){let{content:i,specVersion:s}=await oN(e,t,r,n);return{content:new Ar({mapsAsObjects:!1,int64AsType:"bigint"}).unpack(i),specVersion:s,estimatedSize:e.byteLength*3}}var Y1=Je(oh([Is(Uint8Array)]),Ye(e=>e[0])),Ns=Je(Y1,bs(12),Ye(wh)),aN=Je(Y1,bs(8),Ye(wh)),cN=BigInt(Number.MIN_SAFE_INTEGER),lN=BigInt(Number.MAX_SAFE_INTEGER),uN=Je(rh(),eh(e=>e>=cN&&e<=lN,`Number outside supported range: [${Number.MIN_SAFE_INTEGER}, ${Number.MAX_SAFE_INTEGER}]`),Ye(Number)),$t=Hr([uN,Je(nh(),th())]);function mt(e){let t=Object.keys(e);return Je(ot(_t()),bs(t.length),Ye(r=>Object.fromEntries(t.map((n,i)=>[n,r[i]]))),ur(e))}var Ds=Ns,J1=Ns,xc=aN;function Sc(e,t,r){try{return{...g1(e,r.content),estimatedSize:r.estimatedSize}}catch(n){throw p1(n)?new Error(`Error parsing icechunk ${t}: ${JSON.stringify(d1(n.issues))}`):n}}var fN=2,hN=ur({Inline:Is(Uint8Array)}),pN=_t(),dN=Ue(),mN=mt({location:dN,offset:$t,length:$t,chunksum:pN}),gN=ur({Virtual:mN}),yN=mt({id:J1,offset:$t,length:$t}),vN=ur({Ref:yN}),xN=Je(Xt(Ue(),_t()),Ye(Object.fromEntries),Hr([hN,gN,vN])),SN=mt({id:Ds,chunks:Xt(xc,Xt(Je(ot($t),Ye(e=>e.join())),xN))});async function H1(e,t){let r=await vc(e,yc,fN,t);return Sc(SN,"chunk manifest",r)}function W1(e,t){return Ve(e,`manifests/${t}`)}function X1(e){if(Ee(e),Object.keys(e).length!==1)throw new Error(`Expected object with only a "snapshot" property, but received: ${JSON.stringify(e)}`);let t=re(e,"snapshot",et);if(!Ps(t))throw new Error(`Expected icechunk snapshot id but received: ${JSON.stringify(t)}`);return t}function Ps(e){return e.match(/^[0-9ABCDEFGHJKMNPQRSTVWXYZ]{20}$/)!==null}function Z1(e){return e.match(/^[0-9ABCDEFGHJKMNPQRSTVWXYZ]{8}\.json$/)!==null}var wN=1,Q1=Ns,EN=Ns,bN=mt({id:Ds,sizeBytes:$t,numRows:$t}),IN=mt({id:EN}),_N=Je(Xt(Ue(),_t()),Ye(Object.fromEntries),Hr([ur({Inline:_t()})])),CN=lc(["Slash","Dot"]),tS=Xt(Ue(),_t()),AN=mt({name:Ue(),configuration:tS}),MN=Je(Xt(Ue(),_t()),Ye(e=>{let t=Array.from(e.values());if(t.length!==1)throw new Error(`Expected a single key, but received: ${JSON.stringify(Array.from(e.keys()))}`);return t[0]})),kN=mt({name:Ue(),configuration:tS}),TN=ot(_s(Ue())),NN=mt({shape:ot($t),dataType:Ue(),chunkShape:ot($t),chunkKeyEncoding:CN,fillValue:MN,codecs:ot(AN),storageTransformers:ot(kN),dimensionNames:_s(TN)}),eS=ot($t),DN=sh([eS,eS]),PN=mt({objectId:Ds,extents:DN}),RN=lc(["Group"]),ON=ur({Array:mt({metadata:NN,manifests:ot(PN)})}),UN=Hr([RN,Je(Xt(Ue(),_t()),Ye(Object.fromEntries),ON)]),LN=mt({id:xc,path:Je(Ue(),Ye(e=>e==="/"?"":e.slice(1)+"/")),userAttributes:_N,nodeData:UN}),FN=Je(Xt(Ue(),LN),Ye(e=>Array.from(e.values()).sort((t,r)=>St(t.path,r.path)))),BN=mt({id:Q1,parentId:_s(Q1),flushedAt:Ue(),message:Ue(),metadata:ih(Ue(),_t()),manifestFiles:Je(ot(bN),Ye(e=>{let t=new Map;for(let r of e)t.set(r.id,r);return t})),attributeFiles:ot(IN),nodes:FN});async function rS(e,t){let r=await vc(e,yc,wN,t);return Sc(BN,"snapshot",r)}function nS(e){let{userAttributes:t,nodeData:r}=e,n;t===null?n=new Map:n=t.Inline;let i=r!=="Group"?zN(r.Array.metadata,n):{zarr_format:3,node_type:"group",attributes:n};return JSON.stringify(i,(s,o)=>o instanceof Map?Object.fromEntries(o):o)}function zN(e,t){let{shape:r,chunkShape:n,chunkKeyEncoding:i,dataType:s,fillValue:o,codecs:a,storageTransformers:c,dimensionNames:l}=e;return{zarr_format:3,node_type:"array",shape:r,data_type:s,chunk_grid:{name:"regular",configuration:{chunk_shape:n}},chunk_key_encoding:{name:"default",configuration:{separator:i==="Dot"?".":"/"}},fill_value:o,codecs:a,storage_transformers:c,dimension_names:l??void 0,attributes:t}}function iS(e,t){let{nodes:r}=e,n=er(r,t,(i,s)=>St(i,s.path));if(n<0)throw new Error(`Node not found: ${JSON.stringify(t)}`);return r[n]}function sS(e,t){return Ve(e,`snapshots/${t}`)}var oS=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),aS=e=>{throw TypeError(e)},cS=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&aS("Object expected");var n,i;r&&(n=t[oS("asyncDispose")]),n===void 0&&(n=t[oS("dispose")],r&&(i=n)),typeof n!="function"&&aS("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},lS=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,l){return l=Error(c),l.name="SuppressedError",l.error=o,l.suppressed=a,l},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};function Eh(e,t,r){let n=new qe(e.chunkManager.addRef(),{get:async(i,s)=>{let o=await e.kvStoreContext.read(i,{...s,throwIfMissing:!0});try{return await r(o.response,s.signal)}catch(a){throw new Error(`Error reading icechunk ${t} from ${i}`,{cause:a})}}});return n.registerDisposer(e.addRef()),n}function uS(e,t,r,n){return e.chunkManager.memoize.get("icechunk:snapshot",()=>Eh(e,"snapshot",async(s,o)=>{let a=await rS(await s.arrayBuffer(),o);return{data:a,size:a.estimatedSize}})).get(sS(t,r),n)}function fS(e,t,r,n){return e.chunkManager.memoize.get("icechunk:manifest",()=>Eh(e,"manifest",async(s,o)=>{let a=await H1(await s.arrayBuffer(),o);return{data:a,size:a.estimatedSize}})).get(W1(t,r),n)}function hS(e,t,r){return e.chunkManager.memoize.get("icechunk:ref",()=>Eh(e,"ref",async i=>({data:X1(await i.json()),size:0}))).get(t,r)}function $N(e,t,r){return e.chunkManager.memoize.get("icechunk:branch",()=>{let i=new qe(e.chunkManager.addRef(),{get:async(s,o)=>{var a=[];try{let u=cS(a,new He(o.progressListener,{message:`Resolving icechunk branch at ${s}`}));try{let h=(await e.kvStoreContext.list(s,{...o,responseKeys:"suffix"})).entries.find(d=>Z1(d.key));if(h===void 0)throw new Error("Failed to find any refs");return{data:await hS(e,Ve(s,h.key),o),size:0}}catch(f){throw new Error(`Error resolving icechunk branch at ${s}`,{cause:f})}}catch(u){var c=u,l=!0}finally{lS(a,c,l)}}});return i.registerDisposer(e.addRef()),i}).get(t,r)}function VN(e,t,r){return e.chunkManager.memoize.get("icechunk:tag",()=>{let i=new qe(e.chunkManager.addRef(),{get:async(s,o)=>{var a=[];try{let u=cS(a,new He(o.progressListener,{message:`Resolving icechunk tag at ${s}`}));try{let[f,h]=await Promise.all([hS(e,Ve(s,"ref.json"),o),e.kvStoreContext.stat(Ve(s,"ref.json.deleted"),o)]);if(h!==void 0)throw new Error("Tag is marked as deleted");return{data:f,size:0}}catch(f){throw new Error(`Error resolving icechunk tag at ${s}`,{cause:f})}}catch(u){var c=u,l=!0}finally{lS(a,c,l)}}});return i.registerDisposer(e.addRef()),i}).get(t,r)}function pS(e,t,r,n){return"snapshot"in r?Promise.resolve(r.snapshot):"branch"in r?$N(e,Ve(t,`refs/branch.${r.branch}/`),n):VN(e,Ve(t,`refs/tag.${r.tag}/`),n)}function dS(e,t){let r,n,i=t.match(/(?:^|\/)(zarr\.json)$/);if(i!==null)r=t.slice(0,-i[1].length);else{let l=t.match(/c(?:[./][0-9]+)*$/);if(l===null)return;r=t.slice(0,-l[0].length);let u=l[0].split(/[./]/),f=u.length-1;n=new Array(f);for(let h=0;h<f;++h)n[h]=Number(u[h+1])}let s=iS(e,r);if(n===void 0)return{node:s};if(s.nodeData==="Group")return;let{shape:o,chunkShape:a}=s.nodeData.Array.metadata,c=o.length;if(c===n.length){for(let l=0;l<c;++l)if(n[l]*a[l]>=o[l])return;return{node:s,chunk:n}}}function jN([e,t],r){for(let n=0,i=r.length;n<i;++n){let s=r[n];if(s<e[n]||s>=t[n])return!1}return!0}async function mS(e,t,r,n,i){let{manifests:s}=r.nodeData.Array,o=n.join(),a=r.id;for(let c of s){if(!jN(c.extents,n))continue;let u=(await fS(e,t,c.objectId,i)).chunks.get(a);if(u===void 0)continue;let f=u.get(o);if(f!==void 0)return f}}async function gS(e,t,r,n,i){let s=dS(r,n);if(s===void 0)return;let{node:o,chunk:a}=s;if(a===void 0)return{totalSize:void 0};let c=await mS(e,t,o,a,i);if(c===void 0)return;let l;return"Inline"in c?l=c.Inline.length:"Virtual"in c?l=c.Virtual.length:l=c.Ref.length,{totalSize:l}}async function GN(e,t,r,n){if("Inline"in r)return us(r.Inline,n.byteRange);let i,s,o;if("Virtual"in r)({location:o,offset:i,length:s}=r.Virtual);else{let{Ref:a}=r;({offset:i,length:s}=a),o=qN(t,a.id)}return new tt(e.kvStoreContext.getFileHandle(o),{offset:i,length:s}).read(n)}function qN(e,t){return Ve(e,`chunks/${t}`)}async function yS(e,t,r,n,i){let s=dS(r,n);if(s===void 0)return;let{node:o,chunk:a}=s;if(a===void 0){let l=nS(o),u=new TextEncoder().encode(l);return us(u,i.byteRange)}let c=await mS(e,t,o,a,i);if(c!==void 0)return GN(e,t,c,i)}var bh="branch.",Ih="tag.";function xS(e,t){let{baseUrl:r,refSpec:n}=e,i=n===void 0?"":`@${KN(n)}/`;return r+`|icechunk:${i}${Ze(t)}`}function KN(e){return"branch"in e?bh+Ze(e.branch):"tag"in e?Ih+Ze(e.tag):e.snapshot}function vS(e){return e.length>0&&!e.includes("/")}function _h(e){if(e!==void 0){if(e.startsWith(bh)){let t=e.substring(bh.length);if(!vS(t))throw new Error(`Invalid branch name: ${JSON.stringify(t)}`);return{branch:decodeURIComponent(t)}}if(e.startsWith(Ih)){let t=e.substring(Ih.length);if(!vS(t))throw new Error(`Invalid tag name: ${JSON.stringify(t)}`);return{tag:decodeURIComponent(t)}}if(Ps(e))return{snapshot:e};throw new Error(`Invalid ref spec: ${JSON.stringify(e)}`)}}function SS(e,t){Rr(e);try{let r=(e.suffix??"").match(/^(?:@([^/]*)(?:\/|$))?(.*)$/),[,n,i]=r;return{baseUrl:t.store.getUrl(nn(t.path)),version:_h(n),path:decodeURIComponent(i)}}catch(r){throw new Error(`Invalid URL: ${e.url}`,{cause:r})}}var wc=class{constructor(t,r,n){this.sharedKvStoreContext=t,this.baseUrl=r,this.refSpec=n}snapshot;async getSnapshot(t){let{snapshot:r}=this;if(r===void 0){let n=await pS(this.sharedKvStoreContext,this.baseUrl,this.refSpec??{branch:"main"},t);r=this.snapshot=await uS(this.sharedKvStoreContext,this.baseUrl,n,t)}return r}getUrl(t){return xS(this,t)}async stat(t,r){let n=await this.getSnapshot(r);return gS(this.sharedKvStoreContext,this.baseUrl,n,t,r)}async read(t,r){let n=await this.getSnapshot(r);return yS(this.sharedKvStoreContext,this.baseUrl,n,t,r)}async list(t,r){let n=await this.getSnapshot(r);return a1(n,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};async function wS(e,t){let{url:r}=t,n=r.suffix??"";if(n==="")return{offset:0,completions:[{value:"@",description:"Ref specifier"}]};let i=n.match(/^@([^/]*)((?:\/|$).*)/);if(i===null)return;let[,s,o]=i;if(o!==""){_h(s);return}let a;if(s.match(/^(?:(?:(?:t|$)(?:a|$)(?:g|$)(?:\.|$))|(?:(?:b|$)(?:r|$)(?:a|$)(?:n|$)(?:c|$)(?:h|$)(?:\.|$)))/)){let l=eu(t.base.path,"refs/");a=Xi(t.base.store,l+decodeURIComponent(s),{signal:t.signal,progressListener:t.progressListener}).then(({directories:u})=>u.map(f=>{let h=f.slice(l.length);return{value:Ze(h)+"/",description:h.startsWith("tag.")?"Tag":"Branch"}}))}let c;if(s.match(/^[0-9ABCDEFGHJKMNPQRSTVWXYZ]{0,20}$/)){let l=eu(t.base.path,"snapshots/");c=Xi(t.base.store,l+s,{signal:t.signal,progressListener:t.progressListener}).then(({entries:u})=>{let f=[];for(let{key:h}of u){let p=h.slice(l.length);Ps(p)&&f.push({value:p+"/",description:"Snapshot"})}return f})}return{offset:1,completions:[...await a??[],...await c??[]]}}function YN(e){return{scheme:"icechunk",description:"Icechunk repository",getKvStore(t,r){let{baseUrl:n,version:i,path:s}=SS(t,r);return{store:new wc(e,n,i),path:s}},completeUrl(t){return wS(e,t)}}}wt.registerKvStoreAdapterProvider(YN);var Ch="middleauth+";function JN(e,t){return e.getCredentialsProvider("middleauthapp",new URL(t).origin)}function HN(e,t,r){return{scheme:Ch+e,description:`${e} with middleauth`,getKvStore(n){let i=n.url.substring(Ch.length),s=JN(t.credentialsManager,i);try{let{baseUrl:o,path:a}=Dn(i);return{store:new r(t,o,Ch+o,ea(s)),path:a}}catch(o){throw new Error(`Invalid URL ${JSON.stringify(n.url)}`,{cause:o})}}}}function ES(e,t){for(let r of["https"])e.registerBaseKvStoreProvider(n=>HN(r,n,t))}ES(wt,wi);function WN(e,t,r){return Ss?e.getCredentialsProvider("gcs",{bucket:r}):e.getCredentialsProvider("ngauth_gcs",{authServer:t,bucket:r})}var bS="gs+ngauth+";function XN(e,t){return{scheme:e,description:Ss?"Google Cloud Storage":"Google Cloud Storage (ngauth)",getKvStore(r){let n=(r.suffix??"").match(/^\/\/([^/]+)\/([^/]+)(\/.*)?$/);if(n===null)throw new Error(`Invalid URL, expected ${r.scheme}://<ngauth-server>/<bucket>/<path>`);let[,i,s,o]=n,a=r.scheme.substring(bS.length)+"://"+i,c=WN(t.credentialsManager,a,s);return{store:new Si(s,`${r.scheme}://${i}/${s}/`,ea(c)),path:decodeURIComponent((o??"").substring(1))}}}}for(let e of["http","https"])Wt.registerBaseKvStoreProvider(t=>XN(`${bS}${e}`,t));var MS=Ri(_S(),1);function CS(e,t){let r=0,n=0;for(let i=t,s=e.byteLength;i<s;++i){let o=e.getUint8(i);if(r+=(o&127)<<n,(o&128)===0){if(r>Number.MAX_SAFE_INTEGER)throw new Error(`Value exceeded ${Number.MAX_SAFE_INTEGER}`);return{offset:i+1,value:r}}n+=7}throw new Error("Unexpected EOF")}function AS(e,t){let r=0n,n=0n;for(let i=t,s=e.byteLength;i<s;++i){let o=e.getUint8(i);if(r|=BigInt(o&127)<<BigInt(n),(o&128)===0)return{offset:i+1,value:r};n+=7n}throw new Error("Unexpected EOF")}var Ec=(e=>(e[e.UNCOMPRESSED=0]="UNCOMPRESSED",e[e.ZSTD=1]="ZSTD",e))(Ec||{});async function Ii(e,t,r,n){if(e.byteLength<18)throw new Error("Unexpected EOF");let i=new DataView(e),s=i.getUint32(0,!1);if(s!==t)throw new Error(`Expected magic value 0x${t.toString(16)} but received 0x${s.toString(16)}`);let o=i.getBigUint64(4,!0);if(o!=BigInt(e.byteLength))throw new Error(`Expected length ${e.byteLength} but received: ${o}`);let a=i.getUint32(e.byteLength-4,!0),c=(0,MS.buf)(new Uint8Array(e,0,e.byteLength-4))>>>0;if(a!=c)throw new Error(`Expected CRC32c checksum of ${a}, but received ${c}`);let l=i.getUint8(12);if(l>r)throw new Error(`Expected version to be <= ${r}, but received: ${l}`);let u=i.getUint8(13),f=new Uint8Array(e,14,e.byteLength-14-4);switch(u){case 0:break;case 1:f=await ue(qr,n,[e],f);break;default:throw new Error(`Unknown compression format ${u}`)}return{reader:{offset:0,data:new DataView(f.buffer,f.byteOffset,f.byteLength)},version:l}}function hr(e,t){let{offset:r,data:n}=e;if(r+t>n.byteLength)throw new Error("Unexpected EOF");return e.offset+=t,new Uint8Array(n.buffer,n.byteOffset+r,t)}function Zt(e){let{value:t,offset:r}=CS(e.data,e.offset);return e.offset=r,t}function Ct(e){let{value:t,offset:r}=AS(e.data,e.offset);return e.offset=r,t}function xn(e,t){let r=Zt(e);if(r>t)throw new Error(`Expected value <= ${t}, but received: ${r}`);return r}function Sn(e){let{offset:t,data:r}=e;if(t+1>r.byteLength)throw new Error("Unexpected EOF");return e.offset+=1,r.getUint8(t)}function kS(e){let{offset:t,data:r}=e;if(t+4>r.byteLength)throw new Error("Unexpected EOF");return e.offset+=4,r.getInt32(t,!0)}function Mh(e){let{offset:t,data:r}=e;if(t+8>r.byteLength)throw new Error("Unexpected EOF");return e.offset+=8,r.getBigUint64(t,!0)}function TS(e){if(e.offset!==e.data.byteLength)throw new Error(`Expected EOF at byte ${e.offset}`)}function Be(e){return(t,r,n)=>{let i=[];for(let s=0;s<r;++s)i[s]=e(t,n);return i}}function bc(e,t){let r=Object.keys(t),n=[];for(let i=0;i<e;++i){let s=Object.fromEntries(r.map(o=>[o,t[o][i]]));n[i]=s}return n}function Zr(e,t){return(r,n,i)=>{let s=Object.fromEntries(Object.entries(e).map(([a,c])=>[a,c(r,n,i)])),o=bc(n,s);if(t!==void 0)for(let a=0;a<n;++a)t?.(o[a],i);return o}}var pr=new Uint8Array(0);function Mr(e,t){let r=Math.min(e.length,t.length);for(let n=0;n<r;++n){let i=e[n]-t[n];if(i!==0)return i}return e.length-t.length}function Rs(e,t){let r=Math.min(e.length,t.length);for(let n=0;n<r;++n){let i=e[n]-t[n];if(i!==0)return{offset:n,difference:i}}return{offset:r,difference:e.length-t.length}}var x7={inclusiveMin:pr,exclusiveMax:Uint8Array.of(0)};function _i(...e){let t=0;for(let i of e)t+=i.length;let r=new Uint8Array(t),n=0;for(let i of e)r.set(i,n),n+=i.length;return r}function Ic(e,t){return e.length>=t.length&&Rs(e,t).offset===t.length}function Th(e,t){let{dataFileTable:r}=t,n=Zt(e);if(n>=r.length)throw new Error(`Invalid data file index ${n}, expected value <= ${r.length}`);return r[n]}var _c=Zr({dataFile:Be(Th),offset:Be(Ct),length:Be(Ct)},(e,t)=>{if(wn(e)){if(t.allowMissing!==!0)throw new Error("Reference to missing value not allowed")}else if(e.offset+e.length>BigInt(Number.MAX_SAFE_INTEGER))throw new Error(`Offset=${e.offset} + length=${e.length} exceeds maximum of ${Number.MAX_SAFE_INTEGER}`)});function wn(e){return e.offset===0xffffffffffffffffn&&e.length===0xffffffffffffffffn}var kh=65535;function Ci(e,t){let r=Zt(e),n=new Uint16Array(r*3);for(let c=1,l=r*3;c<l;++c)n[c]=xn(e,kh);let i=[],s=pr,o=pr,a=new TextDecoder("utf-8",{fatal:!0});for(let c=0;c<r;++c){let l=n[c],u=n[c+r],f=n[c+2*r],h=l+u;if(h>kh)throw new Error(`path_length[${c} = prefix_length(${l}) + suffix_length(${u}) = ${h} > ${kh}`);if(f>h)throw new Error(`base_path_length[${c}] = ${f} > path_length(${h}) = prefix_length(${l}) + suffix_length(${u})`);if(l>Math.min(s.length,f)&&f!==s.length)throw new Error(`path_prefix_length[${c-1}] = ${l} > min(base_path_length[${c-1}] = ${s.length}, base_path_length[${c}] = ${f}) is not valid if base_path_length[${c-1}] != base_path_length[${c}]`);let p=l+u-f,d,m;if(f===0)d=t,s=pr;else if(l>=f)d=i[c-1].baseUrl;else{let g=new Uint8Array(f),y=0,I=Math.max(f-l,0);if(l>0){let _=Math.min(l,f);g.set(s.subarray(0,_)),y=_,l-=_}I!==0&&(g.set(hr(e,I),y),u-=I),d=Ve(t,a.decode(g)),s=g}if(p===0)m="",o=pr;else if(u===0&&p===o.length)m=i[c-1].relativePath;else{let g=new Uint8Array(p),y=0;l!==0&&(g.set(o.subarray(0,l),0),y+=l),u>0&&g.set(hr(e,u),y),m=a.decode(g),o=g}i[c]={baseUrl:d,relativePath:m}}return i}var ZN=215687390,QN=0,NS=1024*1024;async function RS(e,t,r){try{let{reader:n}=await Ii(e,ZN,QN,r),i=Sn(n),s=Ci(n,t),o=Zt(n);if(o===0)throw new Error("Empty b+tree node");if(o>NS)throw new Error(`B+tree node has arity ${o}, which exceeds limit of ${NS}`);return{height:i,...i===0?tD(n,s,o):rD(n,s,o),estimatedSize:n.data.byteLength*3}}catch(n){throw new Error("Error decoding OCDBT b+tree node",{cause:n})}}var Nh=65535;function DS(e){return xn(e,Nh)}function OS(e,t,r){let n=new Uint16Array(t*2);for(let c=1,l=n.length;c<l;++c)n[c]=DS(e);let i=n[t];for(let c=1;c<t;++c)i=Math.min(i,n[c]);let s;if(r){s=new Uint16Array(t);for(let c=0;c<t;++c){let l=s[c]=DS(e);i=Math.min(i,l)}}i=Math.min(n[t],i);for(let c=0,l=0;c<t;++c){let u=n[c];if(u>l)throw new Error(`Child ${c}: Prefix length of ${u} exceeds previous key length ${l}`);let f=n[c+t],h=u+f;if(h>Nh)throw new Error(`Child ${c}: Key length ${h} exceeds limit of ${Nh}`);if(r){let p=s[c];if(p>h)throw new Error(`Child ${c}: subtree common prefix length of ${p} exceeds key length of ${h}`);s[c]-=i}l=h}let o=new Array(t),a;{let c=n[t],l=hr(e,c);a=l.slice(0,i),o[0]=l.slice(i)}for(let c=1;c<t;++c){let l=n[c]-i,u=n[c+t],f=hr(e,u),h=o[c-1];if(Mr(h.subarray(l),f)>=0)throw new Error("Invalid key order");let p=new Uint8Array(l+u);p.set(h.subarray(0,l)),p.set(f,l),o[c]=p}return{keys:o,subtreeCommonPrefixLengths:s,commonPrefix:a}}var PS=1024*1024;function eD(e,t,r){let n=Be(Ct)(e,r,{}),i=hr(e,r);for(let o=0;o<r;++o){let a=i[o];if(a>1)throw new Error(`value_kind[${o}]=${a} is outside valid range [0, 1]`);if(a===0){let c=n[o];if(c>BigInt(PS))throw new Error(`value_length[${o}]=${c} exceeds maximum of ${PS} for an inline value`)}}let s=new Array(r);for(let o=0;o<r;++o){if(i[o]!==1)continue;let a=Th(e,{dataFileTable:t});s[o]={dataFile:a,offset:0n,length:n[o]}}for(let o=0;o<r;++o){if(i[o]!==1)continue;let a=Ct(e);s[o].offset=a}for(let o=0;o<r;++o)i[o]===0&&(s[o]=hr(e,Number(n[o])));return s}function tD(e,t,r){let{keys:n,commonPrefix:i}=OS(e,r,!1),s=eD(e,t,r);return{keyPrefix:i,entries:bc(r,{key:n,value:s})}}function rD(e,t,r){let{keys:n,commonPrefix:i,subtreeCommonPrefixLengths:s}=OS(e,r,!0),o=Dh(e,r,{dataFileTable:t});return{keyPrefix:i,entries:bc(r,{key:n,subtreeCommonPrefixLength:s,node:o})}}var nD=Zr({numKeys:Be(Ct),numTreeBytes:Be(Ct),numIndirectValueBytes:Be(Ct)}),Dh=Zr({location:_c,statistics:nD});function Cc(e,t,r){if(e.height!==t)throw new Error(`Expected height of ${t} but received ${e.height}`);let{keyPrefix:n}=e;if(r.length<n.length){if(Mr(n,r)>=0)return}else if(Mr(n,r.subarray(0,n.length))>=0&&Mr(e.entries[0].key,r.subarray(n.length))>=0)return;throw new Error(`First key [${n}]+[${e.entries[0].key}] < inclusive_min [${r}] specified by parent node`)}function iD(e,t){let r=we(0,e.length,n=>Mr(e[n].key,t)>0);return Math.max(0,r-1)}function sD(e,t){return we(0,e.length,r=>Mr(e[r].key,t)>=0)}function US(e,t){let r=iD(e,t),n=Os(e,r,e.length,t);return[r,n]}function Os(e,t,r,n){return t===r||n.length===0?r:we(t,r,i=>{let{offset:s,difference:o}=Rs(n,e[i].key);return o<0&&s<n.length})}function LS(e,t){let r=sD(e,t),n=Os(e,r,e.length,t);return[r,n]}function FS(e,t){let r=er(e,t,(n,i)=>Mr(n,i.key));if(!(r<0))return e[r]}function BS(e,t){let r=we(0,e.length,s=>Mr(e[s].key,t)>0);if(r===0)return;let n=e[r-1],{subtreeCommonPrefixLength:i}=n;if(!(i!==0&&!Ic(t,n.key.subarray(0,i))))return n}var zS=16;function Ph(e,t,r){let n=2**t,i=xn(e,n),s=pD(e,i,{allowMissing:!0,dataFileTable:r});return aD(s,t),s}function oD(e,t,r,n){let i=Ac(t);if(n>i)throw new Error(`height=${n} exceeds maximum of ${i} for version_tree_arity_log2=${t}`);let s=2**t,o=Uh(e,r,s,n-1);return cD(o,t,n),o}function aD(e,t){let r=2**t;if(e.length===0||e.length>r)throw new Error(`num_children=${e.length} outside valid range [1, ${r}]`);for(let[o,a]of e.entries()){if(wn(a.root.location)){if(a.rootHeight!==0)throw new Error(`non-zero root_height=${a.rootHeight} for empty generation ${a.generationNumber}`);let{statistics:c}=a.root;if(c.numKeys!==0n||c.numTreeBytes!==0n||c.numIndirectValueBytes!==0n)throw new Error(`non-zero statistics for empty generation_number[${o}]=${a.generationNumber}`)}if(a.generationNumber===0n)throw new Error(`generation_number[${o}] must be non-zero`);if(o!==0&&a.generationNumber<=e[o-1].generationNumber)throw new Error(`generation_number[${o}]=${a.generationNumber} <= generation_number[${o-1}]=${e[o-1].generationNumber}`)}let n=e.at(-1).generationNumber,i=e[0].generationNumber,s=Rh(t,0,n);if(i<s)throw new Error(`Generation range [${i}, ${n}] exceeds maximum of [${s}, ${n}]`)}function cD(e,t,r){let n=2**t;if(e.length===0||e.length>n)throw new Error(`num_children=${e.length} outside valid range [1, ${n}]`);let i=1n<<BigInt(t*r);for(let[a,c]of e.entries()){if(c.generationNumber===0n)throw new Error(`generation_number[${a}] must be non-zero`);if(a!==0){let l=e[a-1];if(c.generationNumber<=l.generationNumber)throw new Error(`generation_number[${a}]=${c.generationNumber} >= generation_number[${a-1}]=${l.generationNumber}`);if((c.generationNumber-1n)/i===(l.generationNumber-1n)/i)throw new Error(`generation_number[${a}]=${c.generationNumber} should be in the same child node as generation_number[${a-1}]=${l.generationNumber}`)}if(c.generationNumber%i!==0n)throw new Error(`generation_number[${a}]=${c.generationNumber} is not a multiple of ${i}`);if(c.numGenerations>i)throw new Error(`num_generations[${a}]=${c.numGenerations} for generation_number=${c.generationNumber} is greater than ${i}`)}let s=1n<<BigInt(t),o=e.at(-1);if((o.generationNumber-1n)/i/s!==(e[0].generationNumber-1n)/i/s)throw new Error(`generation_number[0]=${e[0].generationNumber} cannot be in the same node as generation_number[${e.length-1}]=${o.generationNumber}`)}function Rh(e,t,r){return r-(r-1n)%(1n<<BigInt(e*(t+1)))}function Oh(e){let t=Sn(e);if(t===0||t>zS)throw new Error(`Expected version_tree_arity_log2 in range [1, ${zS}] but received: ${t}`);return t}var lD=215683636,uD=0;async function $S(e,t,r){try{let{reader:n}=await Ii(e,lD,uD,r),i=Oh(n),s=Sn(n),o=Ci(n,t);return{versionTreeArityLog2:i,height:s,entries:s===0?Ph(n,i,o):oD(n,i,o,s),estimatedSize:n.data.byteLength*3}}catch(n){throw new Error("Error decoding OCDBT version tree node",{cause:n})}}var fD=Zr({generationNumber:Be(Ct),location:_c,numGenerations:Be(Ct),commitTime:Be(Mh),height:Be((e,{height:t})=>t===void 0?Sn(e):t),cumulativeNumGenerations:Be(()=>0n)});function hD(e){let t=0n;for(let r of e)t+=r.numGenerations,r.cumulativeNumGenerations=t}function Uh(e,t,r,n){let i=xn(e,r),s=fD(e,i,{dataFileTable:t,height:n});return hD(s),s}function Ac(e){return Math.floor(63/e)-1}var pD=Zr({generationNumber:Be(Ct),rootHeight:Be(Sn),root:Dh,commitTime:Be(Mh)});function Lh(e,t){return"generationNumber"in e?mr(e.generationNumber,t.generationNumber):mr(e.commitTime,t.commitTime)}function VS(e,t,r){if("generationNumber"in r)return dD(t,r.generationNumber);if("generationIndex"in r){let{generationIndex:n}=r;return n-=e,n<0n?-1:n>=BigInt(t.length)?t.length:Number(n)}else return mD(t,r.commitTime)}function dD(e,t){let r=er(e,t,(n,i)=>mr(n,i.generationNumber));return r<0?e.length:r}function mD(e,t){let r=we(0,e.length,n=>e[n].commitTime>t);return r===0?e.length:r-1}function Ai(e,t,r){if("generationIndex"in r){let n=r.generationIndex-e;return n<0n?0:n>BigInt(t.length)?t.length:Number(n)}return we(0,t.length,n=>Lh(r,t[n])<=0)}function jS(e,t,r,n){return"generationIndex"in n?r[GS(r,n.generationIndex-t)]:"generationNumber"in n?gD(e,r,n.generationNumber):yD(r,n.commitTime)}function GS(e,t){return we(0,e.length,r=>e[r].cumulativeNumGenerations>t)}function gD(e,t,r){let n=we(0,t.length,s=>t[s].generationNumber>=r);if(n===t.length)return;let i=t[n];if(!(Rh(e,i.height,i.generationNumber)>r))return i}function yD(e,t){let r=we(0,e.length,n=>e[n].commitTime>t);if(r!==0)return e[r-1]}function Mc(e,t,r,n){return"generationIndex"in n?GS(r,n.generationIndex-t):"generationNumber"in n?vD(e,r,n.generationNumber):qS(r,n.commitTime)}function vD(e,t,r){return we(0,t.length,n=>{let i=t[n];return Rh(e,i.height,i.generationNumber)>=r})}function qS(e,t){let r=we(0,e.length,n=>e[n].commitTime>t);return Math.max(0,r-1)}function kc(e,t,r){return"generationIndex"in r?xD(t,r.generationIndex-e):"generationNumber"in r?SD(t,r.generationNumber):qS(t,r.commitTime)}function xD(e,t){return we(0,e.length,r=>{let n=e[r];return n.cumulativeNumGenerations-n.numGenerations>=t})}function SD(e,t){return we(0,e.length,r=>e[r].generationNumber>=t)}function Tc(e,t,r,n,i){if(e.height!==n)throw new Error(`Expected height of ${n} but received: ${e.height}`);if(e.versionTreeArityLog2!==t.versionTreeArityLog2)throw new Error(`Expected version_tree_arity_log2=${t.versionTreeArityLog2} but received: ${e.versionTreeArityLog2}`);let{generationNumber:s}=e.entries.at(-1);if(s!==r)throw new Error(`Expected generation number ${r} but received: ${s}`);let o=e.height===0?BigInt(e.entries.length):e.entries.at(-1).cumulativeNumGenerations;if(o!==i)throw new Error(`Expected ${i}, but received: ${o}`)}function wD(e){let t=hr(e,16).slice(),r=Zt(e);if(r>1)throw new Error(`Unknown manifest kind: ${r}`);let n=Zt(e),i=Zt(e),s=Oh(e),o=Zt(e),a;switch(o){case Ec.UNCOMPRESSED:break;case Ec.ZSTD:a=kS(e);break;default:throw new Error(`Invalid compression method: ${o}`)}return{uuid:t,manifestKind:r,maxInlineValueBytes:n,maxDecodedNodeBytes:i,versionTreeArityLog2:s,compressionMethod:o,zstdLevel:a}}function ED(e,t,r){let n=Ci(e,r),i=Ph(e,t.versionTreeArityLog2,n),s=bD(e,t.versionTreeArityLog2,n,i.at(-1).generationNumber);return{inlineVersions:i,versionTreeNodes:s,numGenerations:BigInt(i.length)+(s.at(-1)?.cumulativeNumGenerations??0n)}}function bD(e,t,r,n){let i=Ac(t),s=Uh(e,r,i,void 0);return ID(t,n,s),s}function ID(e,t,r){let n=Ac(e);for(let[s,o]of r.entries()){if(o.height===0||o.height>n)throw new Error(`entry_height[${s}]=${o.height} outside valid range [1, ${n}]`);if(o.generationNumber===0n)throw new Error(`generation_number[${s}] must be non-zero`);if(s>0){let a=r[s-1];if(o.generationNumber<=a.generationNumber)throw new Error(`generation_number[${s}]=${o.generationNumber} <= generation_number[${s-1}]=${a.generationNumber}`);if(o.height>=a.height)throw new Error(`entry_height[${s}]=${o.height} >= entry_height[${s-1}]=${a.height}`)}}let i=r.length;for(let{minGenerationNumber:s,maxGenerationNumber:o,height:a}of _D(t,e)){if(i===0)break;let c=r[i-1];if(c.height!==a)continue;--i;let{generationNumber:l}=c;if(l<s||l>o)throw new Error(`generation_number[${i}]=${l} is outside expected range [${s}, ${o}] for height ${a}`)}if(i!==0)throw new Error(`Unexpected child with generation_number[${i-1}]=${r[i-1].generationNumber} and entry_height=${r[i-1].height} given last generation_number=${t}`)}function _D(e,t){e=e-1n>>BigInt(t)<<BigInt(t);let r=1,n=[];for(;e!==0n;){let i=BigInt((r+1)*t),s=e-1n>>i<<i,o=s+1n;n.push({minGenerationNumber:o,maxGenerationNumber:e,height:r}),++r,e=s}return n}var CD=215693866,AD=0;async function KS(e,t,r){try{let{reader:n}=await Ii(e,CD,AD,r),i=wD(n),s=i.manifestKind===0?ED(n,i,t):void 0;return TS(n),{config:i,versionTree:s,estimatedSize:n.data.byteLength*3}}catch(n){throw new Error("Error decoding OCDBT manifest",{cause:n})}}var YS=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),JS=e=>{throw TypeError(e)},MD=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&JS("Object expected");var n,i;r&&(n=t[YS("asyncDispose")]),n===void 0&&(n=t[YS("dispose")],r&&(i=n)),typeof n!="function"&&JS("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},kD=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,l){return l=Error(c),l.name="SuppressedError",l.error=o,l.suppressed=a,l},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};function TD(e,t,r){return e.chunkManager.memoize.get("ocdbt:manifest",()=>{let i=new qe(e.chunkManager.addRef(),{get:async(s,o)=>{var a=[];try{let u=Ve(s.baseUrl,s.relativePath),f=MD(a,new He(o.progressListener,{message:`Reading OCDBT manifest from ${u}`})),h=await e.kvStoreContext.read(u,{...o,throwIfMissing:!0});try{let p=await KS(await h.response.arrayBuffer(),s.baseUrl,o.signal);return{data:p,size:p.estimatedSize}}catch(p){throw new Error(`Error reading OCDBT manifest from ${u}`,{cause:p})}}catch(u){var c=u,l=!0}finally{kD(a,c,l)}}});return i.registerDisposer(e.addRef()),i}).get(t,r)}async function Us(e,t,r){let n=await TD(e,{baseUrl:t,relativePath:"manifest.ocdbt"},r);if(n.versionTree===void 0)throw new Error("only manifest_kind=single is supported");return n}function HS(e,t,r){let n=new qe(e.chunkManager.addRef(),{get:async(i,s)=>{let{dataFile:o}=i,a=Ve(o.baseUrl,o.relativePath),c=await e.kvStoreContext.read(a,{...s,throwIfMissing:!0,byteRange:{offset:Number(i.offset),length:Number(i.length)}});try{let l=await r(await c.response.arrayBuffer(),o.baseUrl,s.signal);return{data:l,size:l.estimatedSize}}catch(l){throw new Error(`Error reading OCDBT ${t} from ${a}`,{cause:l})}},encodeKey:({dataFile:i,offset:s,length:o})=>JSON.stringify([i,`${s}/${o}`])});return n.registerDisposer(e.addRef()),n}function Nc(e,t,r){return e.chunkManager.memoize.get("ocdbt:btree",()=>HS(e,"b+tree node",RS)).get(t,r)}function Dc(e,t,r){return e.chunkManager.memoize.get("ocdbt:versionnode",()=>HS(e,"version tree node",$S)).get(t,r)}var Fh=!1;async function WS(e,t,r,n){let i=[],s=new Set;wn(t.root.location)||await XS(t.root,t.rootHeight,pr,0,{sharedKvStoreContext:e,prefix:r,entries:i,directories:s,signal:n.signal,progressListener:n.progressListener});let o=Eo({entries:i,directories:Array.from(s)});return Fh&&console.log(JSON.stringify(o)),o}async function XS(e,t,r,n,i){i.signal?.throwIfAborted();let s=await Nc(i.sharedKvStoreContext,e.location,i);Cc(s,t,r.subarray(n));let o=_i(r.subarray(0,n),s.keyPrefix);Fh&&console.log("listSubtree",{nodeReference:e,height:t,inclusiveMinKey:r,subtreeCommonPrefixLength:n});let a=u=>{try{i.directories.add(new TextDecoder("utf-8",{fatal:!0}).decode(u))}catch{}},{prefix:c}=i;{let{offset:u,difference:f}=Rs(c,o);if(f!==0&&u<Math.min(c.length,o.length))return}if(c.length<o.length){let u=o.indexOf(47,c.length);if(u!==-1){a(o.subarray(0,u));return}}let l=c.subarray(o.length);if(s.height>0){let u=s.entries,[f,h]=US(u,l);Fh&&console.log("Got entry range",f,h,u.length,l);let p=[];for(let d=f;d<h;){let m=u[d];++d;let{key:g}=m,{subtreeCommonPrefixLength:y}=m;if(y>l.length){let I=g.indexOf(47,l.length);if(I!==-1){let _=g.subarray(0,I);a(_i(o,_)),d=Os(u,d,h,_);continue}}p.push(XS(m.node,t-1,_i(o,m.key),o.length+m.subtreeCommonPrefixLength,i))}await Promise.all(p)}else{let u=s.entries,[f,h]=LS(u,l);for(let p=f;p<h;){let d=u[p];++p;let{key:m}=d,g=m.indexOf(47,l.length);if(g!==-1){let y=m.subarray(0,g);a(_i(o,y)),p=Os(u,p,h,m.subarray(0,g+1));continue}try{i.entries.push({key:new TextDecoder("utf-8",{fatal:!0}).decode(_i(o,m))})}catch{}}}}var ZS=!1;async function Bh(e,t,r,n){if(!wn(t.root.location))return await ND(e,t.root,t.rootHeight,pr,r,n)}async function ND(e,t,r,n,i,s){for(;;){let o=await Nc(e,t.location,s);if(ZS&&console.log(t,r,o,n,i),Cc(o,r,n),!Ic(i,o.keyPrefix)){ZS&&console.log("not found due to key prefix mismatch",i,o.keyPrefix);return}if(o.height===0)return FS(o.entries,i);let a=BS(o.entries,i);if(a===void 0)return;let{subtreeCommonPrefixLength:c}=a;i=i.subarray(c),t=a.node,n=a.key.subarray(c),--r}}async function QS(e,t,r){let{value:n}=t;if(n instanceof Uint8Array)return us(n,r.byteRange);let{offset:i,length:s,dataFile:{baseUrl:o,relativePath:a}}=n,{store:c,path:l}=e.kvStoreContext.getKvStore(o);return await new tt(new ke(c,l+a),{offset:Number(i),length:Number(s)}).read(r)}function Ls(e){if(e===void 0)return"HEAD";if("generationNumber"in e)return`v${e.generationNumber}`;let{commitTime:t}=e;return Rc(t)}function Pc(e){if(e===void 0)return;let t=e.match(/^(?:v([1-9]\d*)|(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*)?Z))$/);if(t===null)throw new Error(`Invalid OCDBT version specifier: ${JSON.stringify(e)}`);let[,r]=t;if(r!==void 0){let n=BigInt(r);if(n>0xffffffffffffffffn)throw new Error(`Invalid generation number: ${n}`);return{generationNumber:n}}return{commitTime:DD(e)}}function DD(e){let t=e.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:(\.\d*))?Z$/);if(t===null)throw new Error(`Invalid commit timestamp: ${JSON.stringify(e)}`);let[,r,n]=t;return ew(Date.parse(r+"Z"),n)}function ew(e,t){let r=BigInt(e)*1000000n;if(t!==void 0&&t.length>1){let n=Number(t);r+=BigInt(Math.min(999999999,Math.round(n*1e9)))}return r}function Rc(e){let t=e%1000000000n,r=e/1000000000n;t<0n&&(t+=1000000000n,r-=1n);let n=new Date(Number(r)*1e3).toISOString();if(n.length!==24)throw new Error(`Invalid commit time: ${e} -> ${n}`);return n=n.slice(0,19),t!==0n&&(n+="."+t.toString().padStart(9,"0").replace(/0+$/,"")),n+="Z",n}var PD=new RegExp("^(\\d{0,4})(?:(?<=\\d{4})-(\\d{0,2})(?:(?<=\\d{2})-(\\d{0,2})(?:(?<=\\d{2})T(\\d{0,2})(?:(?<=\\d{2}):(\\d{0,2})(?:(?<=\\d{2}):(\\d{0,2})(?:(?<=\\d{2})(\\.\\d*)?(Z)?)?)?)?)?)?)?$");function Mi(e,t,r,n,i){t=t??"";let s=parseInt(t.padEnd(r,"0"),10),o=parseInt(t.padEnd(r,"9"),10);if(s>i)throw new Error(`Invalid ${e} prefix: ${t}`);return[Math.max(n,s),Math.min(i,o)]}function RD(e,t){let r=new Date(0);return r.setUTCFullYear(e),r.setUTCMonth(t),r.setUTCDate(0),r.getUTCDate()}function tw(e){let t=e.match(PD);if(t===null)throw new Error(`Expected prefix of ISO-8601 "YYYY-MM-DDThh:mm:ss.sssssssssZ" format, but received: ${JSON.stringify(e)}`);let r=Mi("year",t[1],4,0,9999),n=Mi("month",t[2],2,1,12),i=RD(r[1],n[1]),s=Mi("day",t[3],2,1,i),o=Mi("hour",t[4],2,0,23),a=Mi("minute",t[5],2,0,59),c=Mi("second",t[6],2,0,59),l=t[7]??".",u=t[8],f=l.padEnd(10,"0"),h=u===void 0?l.padEnd(10,"9"):f,p=[f,h];function d(m){let g=new Date(0);return g.setUTCFullYear(r[m]),g.setUTCMonth(n[m]-1),g.setUTCDate(s[m]),g.setUTCHours(o[m]),g.setUTCMinutes(a[m]),g.setUTCSeconds(c[m]),ew(g.getTime(),p[m])}return[d(0),d(1)]}async function rw(e,t,r,n){return e.chunkManager.memoize.get("ocdbt:version",()=>{let s=new qe(e.chunkManager.addRef(),{get:async({url:o,version:a},c)=>{let l=await Us(e,o,c),u=await OD(e,l,a,n);if(u===void 0)throw new Error(`Version ${Ls(a)} not found`);return{data:u.ref,size:0}},encodeKey:({url:o,version:a})=>{let c;return a!==void 0&&(c=Ls(a)),JSON.stringify([o,c])}});return s.registerDisposer(e.addRef()),s}).get({url:t,version:r},n)}async function OD(e,t,r,n){let{versionTree:i}=t;if(r===void 0){let{versionTreeNodes:a,inlineVersions:c}=i,l=c.length-1;return{ref:c[l],generationIndex:(a.at(-1)?.cumulativeNumGenerations??0n)+BigInt(l)}}let{ref:s,generationIndex:o}=await UD(e,t,r,n);if(s!==void 0)return{ref:s,generationIndex:o}}async function nw(e,t,r,n){let{generationIndex:i}=await LD(e,t,r,n);return i}async function iw(e,t,r,n){let{generationIndex:i}=await FD(e,t,r,n);return i}function zh(e){let{isInline:t,findInLeaf:r,findInInterior:n}=e;async function i(o,a,c,l){let{config:u,versionTree:f}=a,h=f.versionTreeNodes.at(-1)?.cumulativeNumGenerations??0n,{inlineVersions:p}=f;if(t(u,h,p,c)){let g=r(u,h,p,c);return{ref:p[g],generationIndex:h+BigInt(g)}}let{versionTreeNodes:d}=f;if(d.length===0)return{ref:void 0,generationIndex:0n};let m=n(u,0n,d,c);return m===void 0?{ref:void 0,generationIndex:0n}:await s(o,a.config,0n+m.cumulativeNumGenerations-m.numGenerations,m,c,l)}async function s(o,a,c,l,u,f){for(;;){let h=await Dc(o,l.location,f);if(Tc(h,a,l.generationNumber,l.height,l.numGenerations),h.height===0){let d=h.entries,m=r(a,c,d,u);return{ref:d[m],generationIndex:c+BigInt(m)}}let p=n(a,c,h.entries,u);if(p===void 0)return{ref:void 0,generationIndex:c};l=p,c+=l.cumulativeNumGenerations-l.numGenerations}}return i}function $h(e,t,r){return"generationIndex"in r?r.generationIndex>=e:Lh(r,t[0])>=0}var UD=zh({isInline(e,t,r,n){return $h(t,r,n)},findInLeaf(e,t,r,n){return VS(t,r,n)},findInInterior(e,t,r,n){return jS(e.versionTreeArityLog2,t,r,n)}}),LD=zh({isInline(e,t,r,n){return $h(t,r,n)},findInLeaf(e,t,r,n){return Ai(t,r,n)},findInInterior(e,t,r,n){let i=Mc(e.versionTreeArityLog2,t,r,n);return r[i]}}),FD=zh({isInline(e,t,r,n){return $h(t,r,n)},findInLeaf(e,t,r,n){return Ai(t,r,n)},findInInterior(e,t,r,n){let i=kc(t,r,n);return r[i]}});function sw(e,t){let{version:r,baseUrl:n}=e,i=r===void 0?"":`@${Ls(r)}/`;return n+`|ocdbt:${i}${Ze(t)}`}function ow(e,t){Rr(e);try{let r=(e.suffix??"").match(/^(?:@([^/]*)(?:\/|$))?(.*)$/),[,n,i]=r;return{baseUrl:t.store.getUrl(nn(t.path)),version:Pc(n),path:decodeURIComponent(i)}}catch(r){throw new Error(`Invalid URL: ${e.url}`,{cause:r})}}var Oc=class{constructor(t,r,n){this.sharedKvStoreContext=t,this.baseUrl=r,this.version=n}root;async getRoot(t){let{root:r}=this;return r===void 0&&(r=this.root=await rw(this.sharedKvStoreContext,this.baseUrl,this.version,t)),r}getUrl(t){return sw(this,t)}async stat(t,r){let n=await this.getRoot(r),i=new TextEncoder().encode(t),s=await Bh(this.sharedKvStoreContext,n,i,r);if(s===void 0)return;let{value:o}=s;return{totalSize:Number(o.length)}}async read(t,r){let n=await this.getRoot(r),i=new TextEncoder().encode(t),s=await Bh(this.sharedKvStoreContext,n,i,r);if(s!==void 0)return await QS(this.sharedKvStoreContext,s,r)}async list(t,r){let n=await this.getRoot(r),i=new TextEncoder().encode(t);return await WS(this.sharedKvStoreContext,n,i,r)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};var aw=!1;async function Vh(e,t,r){let{inclusiveMin:n,exclusiveMax:i}=r;aw&&console.log("listVersions",n,i);let s=n===void 0?{generationIndex:0n}:n,o=i===void 0?{generationIndex:t.versionTree.numGenerations}:i,{config:a,versionTree:c}=t,{versionTreeArityLog2:l}=a,u,f=[];{let m=c.versionTreeNodes.at(-1)?.cumulativeNumGenerations??0n;h(m,c.inlineVersions),await p(0n,c.versionTreeNodes)}function h(m,g){let y=Ai(m,g,s),I=Ai(m,g,o),_=m+BigInt(y);(u===void 0||_<u)&&(u=_);for(let E=y;E<I;++E)f.push(g[E])}async function p(m,g){r.signal?.throwIfAborted();let y=Mc(l,m,g,s),I=kc(m,g,o);aw&&console.log("listVersions: visitInteriorEntries",s,o,`generationIndex=${m}`,`versionNodes.length=${g.length}`,y,I);let _=[];for(let E=y;E<I;++E){let b=g[E];_.push(d(m+b.cumulativeNumGenerations-b.numGenerations,b))}await Promise.all(_)}async function d(m,g){let y=await Dc(e,g.location,r);Tc(y,a,g.generationNumber,g.height,g.numGenerations),y.height===0?h(m,y.entries):await p(m,y.entries)}return f.sort((m,g)=>mr(m.generationNumber,g.generationNumber)),{generationIndex:u??0n,versions:f}}async function BD(e,t,r,n,i,s){if(n<=r+i){let{versions:c}=await Vh(e,t,{inclusiveMin:{generationIndex:r},exclusiveMax:{generationIndex:n},...s});return c}let[{versions:o},{versions:a}]=await Promise.all([r,n-i/2n].map(c=>Vh(e,t,{inclusiveMin:{generationIndex:c},exclusiveMax:{generationIndex:c+i/2n},...s})));return[...o,...a]}async function cw(e,t){let{url:r}=t,n=r.suffix??"";if(n==="")return{offset:0,completions:[{value:"@",description:"Version specifier"}]};let i=n.match(/^@([^/]*)((?:\/|$).*)/);if(i===null)return;let[,s,o]=i;if(o!==""){Pc(s);return}let{base:a}=t,c=a.store.getUrl(nn(a.path));if(!s.startsWith("v")){let[l,u]=tw(s),f={signal:t.signal,progressListener:t.progressListener},h=await Us(e,c,f),[p,d]=await Promise.all([nw(e,h,{commitTime:l},f),iw(e,h,{commitTime:u+1n},f)]),g=(await BD(e,h,p,d,100n,{signal:t.signal,progressListener:t.progressListener})).map(y=>({value:`${Rc(y.commitTime)}/`,description:`v${y.generationNumber}`}));return g.reverse(),{offset:1,completions:g}}if(s==="v"){let{base:l}=t,f=(await Us(e,l.store.getUrl(l.path),t)).versionTree.inlineVersions.map(h=>({value:`v${h.generationNumber}/`,description:Rc(h.commitTime)}));return f.reverse(),{offset:1,completions:f}}return{offset:1,completions:[{value:`${s}/`}]}}function zD(e){return{scheme:"ocdbt",description:"OCDBT database",getKvStore(t,r){let{baseUrl:n,version:i,path:s}=ow(t,r);return{store:new Oc(e,n,i),path:s}},completeUrl(t){return cw(e,t)}}}wt.registerKvStoreAdapterProvider(zD);var $D=["http://doc.s3.amazonaws.com/2006-03-01/","http://s3.amazonaws.com/doc/2006-03-01/"];function VD(e){return $D.includes(e.namespaceURI)&&e.tagName==="ListBucketResult"}async function Uc(e,t,r,n){let i="/";try{let s=await r(`${e}?list-type=2&prefix=${encodeURIComponent(t)}&delimiter=${encodeURIComponent(i)}&encoding-type=url`,{headers:{accept:"application/xml,text/xml"},signal:n.signal,progressListener:n.progressListener}),o=s.headers.get("content-type");if(o===null||/\b(application\/xml|text\/xml|text\/html)\b/i.exec(o)===null)throw new Error(`Expected XML content-type but received: ${o}`);let a=await s.text(),c=new DOMParser().parseFromString(a,"application/xml"),{documentElement:l}=c;if(!VD(l))throw new Error(`Received unexpected XML root element <${l.tagName} xmlns="${l.namespaceURI}">`);let u=l.namespaceURI,f=()=>u,h=c.evaluate("//CommonPrefixes/Prefix",c,f,XPathResult.UNORDERED_NODE_SNAPSHOT_TYPE,null),p=[];for(let g=0,y=h.snapshotLength;g<y;++g){let I=h.snapshotItem(g).textContent;I!==null&&(I=decodeURIComponent(I),p.push(I.substring(0,I.length-i.length)))}let d=[],m=c.evaluate("//Contents/Key",c,f,XPathResult.UNORDERED_NODE_SNAPSHOT_TYPE,null);for(let g=0,y=m.snapshotLength;g<y;++g){let I=m.snapshotItem(g).textContent;I!==null&&d.push({key:decodeURIComponent(I)})}return{directories:p,entries:d}}catch(s){throw new Error("S3-compatible listing not supported",{cause:s})}}function lw(e,t,r){let{baseUrl:n,path:i}=Dn(e);return Uc(n,i,t,r)}function jD(e){let t=new URL(e),r=t.pathname.match(/^\/([^/]+)(?:\/(.*))$/);if(r===null)return;let[,n,i]=r;return{bucketUrl:`${t.origin}/${n}/${t.search}`,bucket:decodeURIComponent(n),prefix:decodeURIComponent(i)}}async function uw(e,t,r){let n=jD(e);if(n===void 0)throw new Error(`Path-style S3 URL ${JSON.stringify(e)} must specify bucket`);let{bucketUrl:i,bucket:s,prefix:o}=n,a=await Uc(i,o,t,r),c=Ze(s)+"/";return{entries:a.entries.map(l=>({key:c+l.key})),directories:a.directories.map(l=>c+l)}}function GD(e){return e.getUncounted("s3:urlkind",()=>new Map)}async function fw(e,t,r,n,i){let s=GD(r),o=s.get(t);if(o==="virtual")return await lw(e,n,i);if(o==="path")return await uw(e,n,i);if(o!==null)try{let{result:a,urlKind:c}=await Promise.any([lw(e,n,i).then(l=>({result:l,urlKind:"virtual"})),uw(e,n,i).then(l=>({result:l,urlKind:"path"}))]);return s.set(t,c),a}catch(a){throw i.signal?.throwIfAborted(),s.set(t,null),new Error("Neither virtual hosted nor path-style S3 listing supported",{cause:a})}throw new Error("Neither virtual hosted nor path-style S3 listing supported")}var hw=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),pw=e=>{throw TypeError(e)},qD=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&pw("Object expected");var n,i;r&&(n=t[hw("asyncDispose")]),n===void 0&&(n=t[hw("dispose")],r&&(i=n)),typeof n!="function"&&pw("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},KD=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,l){return l=Error(c),l.name="SuppressedError",l.error=o,l.suppressed=a,l},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};var Lc=class{constructor(t,r,n,i,s=Re){this.sharedKvStoreContext=t,this.baseUrl=r,this.baseUrlForDisplay=n,this.knownToBeVirtualHostedStyle=i,this.fetchOkImpl=s}stat(t,r){let n=Ht(this.baseUrl,t);return pn(this,t,n,r,this.fetchOkImpl)}read(t,r){let n=Ht(this.baseUrl,t);return ei(this,t,n,r,this.fetchOkImpl)}list(t,r){var n=[];try{let{progressListener:o}=r,a=qD(n,o===void 0?void 0:new He(o,{message:`Listing prefix ${this.getUrl(t)}`}));return this.knownToBeVirtualHostedStyle?Uc(this.baseUrl,t,this.fetchOkImpl,r):fw(Ht(this.baseUrl,t),this.baseUrlForDisplay,this.sharedKvStoreContext.chunkManager.memoize,this.fetchOkImpl,r)}catch(o){var i=o,s=!0}finally{KD(n,i,s)}}getUrl(t){return Ht(this.baseUrlForDisplay,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function YD(e,t){return{scheme:"s3",description:"S3 (anonymous)",getKvStore(r){let n=(r.suffix??"").match(/^\/\/([^/]+)(\/.*)?$/);if(n===null)throw new Error("Invalid URL, expected `s3://<bucket>/<path>`");let[,i,s]=n;return{store:new t(e,`https://${i}.s3.amazonaws.com/`,`s3://${i}/`,!0),path:decodeURIComponent((s??"").substring(1))}}}}function JD(e,t,r){return{scheme:`s3+${t}`,description:`S3-compatible ${t} server`,getKvStore(n){let i=(n.suffix??"").match(/^\/\/([^/]+)(\/.*)?$/);if(i===null)throw new Error("Invalid URL, expected `s3+${httpScheme}://<host>/<path>`");let[,s,o]=i;return{store:new r(e,`${t}://${s}/`,`s3+${t}://${s}/`,!1),path:decodeURIComponent((o??"").substring(1))}}}}function dw(e,t){e.registerBaseKvStoreProvider(r=>YD(r,t));for(let r of["http","https"])e.registerBaseKvStoreProvider(n=>JD(n,r,t))}var Fc=class extends Lc{list(t,r){return ac(this.sharedKvStoreContext,this.getUrl(t),r)}};dw(wt,Fc);var yw=Ri(gw(),1);var Gh=22,HD=65535,WD=101010256,XD=101075792;function ZD(e){let t=0,r;return async function(i,s,o){if(r!==void 0&&i>t&&i+s<=t+r.length)return r.subarray(i-t,i+s-t);let a=await e(i,s,o);return t=i,r=a,a}}function QD(e){let t=new DataView(e.buffer,e.byteOffset,e.byteLength),r=e.length;for(let n=r-Gh;n>=0;--n){if(t.getUint32(n,!0)!==WD)continue;let i=t.getUint16(n+20,!0),s=r-n-Gh;if(i!==s)continue;let o=t.getUint16(n+4,!0),a=t.getUint16(n+10,!0),c=t.getUint32(n+12,!0),l=t.getUint32(n+16,!0);return{eocdrOffset:n,diskNumber:o,entryCount:a,centralDirectorySize:c,centralDirectoryOffset:l}}}async function e3(e,t,r){let n=Math.min(Gh+HD,t),i=t-n,s=await e(i,n,r),o=QD(s);if(o===void 0)throw new Error("End of central directory record signature not found; either not a zip file or file is truncated.");let{eocdrOffset:a,diskNumber:c,entryCount:l,centralDirectorySize:u,centralDirectoryOffset:f}=o;if(c!==0)throw new Error(`Multi-volume zip files are not supported. This is volume: ${c}`);let h=s.slice(a+22,s.length);return l===65535||f===4294967295?await r3(e,a,h,r):await vw(e,f,u,l,h,r)}var t3=117853008;async function r3(e,t,r,n){let i=t-20,s=await e(i,20,n),o=new DataView(s.buffer,s.byteOffset,s.byteLength);if(o.getUint32(0,!0)!==t3)throw new Error("invalid zip64 end of central directory locator signature");let a=o.getBigUint64(8,!0),c=await e(Number(a),56,n),l=new DataView(c.buffer,c.byteOffset,c.byteLength);if(l.getUint32(0,!0)!==XD)throw new Error("invalid zip64 end of central directory record signature");let u=l.getBigUint64(32,!0),f=l.getBigUint64(40,!0),h=l.getBigUint64(48,!0);return vw(e,Number(h),Number(f),Number(u),r,n)}var n3=33639248;async function vw(e,t,r,n,i,s){let o=0,a=await e(t,r,s),c=[],l=new DataView(a.buffer,a.byteOffset,a.byteLength),u=new TextDecoder;for(let f=0;f<n;++f){let h=l.getUint32(o+0,!0);if(h!==n3)throw new Error(`invalid central directory file header signature: 0x${h.toString(16)}`);let p=l.getUint16(o+4,!0),d=l.getUint16(o+6,!0),m=l.getUint16(o+8,!0),g=l.getUint16(o+10,!0),y=l.getUint16(o+12,!0),I=l.getUint16(o+14,!0),_=l.getUint32(o+16,!0),E=l.getUint32(o+20,!0),b=l.getUint32(o+24,!0),C=l.getUint16(o+28,!0),v=l.getUint16(o+30,!0),w=l.getUint16(o+32,!0),x=l.getUint16(o+36,!0),k=l.getUint32(o+38,!0),M=l.getUint32(o+42,!0);if(m&64)throw new Error("strong encryption is not supported");o+=46;let P=a.subarray(o,o+=C),F=(m&2048)!==0,S=[];for(let L=0;L<v-3;){let G=l.getUint16(o+L+0,!0),V=l.getUint16(o+L+2,!0),Y=L+4,J=Y+V;if(J>v)throw new Error("extra field length exceeds extra field buffer size");S.push({id:G,offset:o+Y,length:V}),L=J}o+=v;let O=a.slice(o,o+=w);if(b===4294967295||E===4294967295||M===4294967295){let L=S.find(J=>J.id===1);if(L===void 0)throw new Error("expected zip64 extended information extra field");let{offset:G,length:V}=L,Y=0;if(b===4294967295){if(Y+8>V)throw new Error("zip64 extended information extra field does not include uncompressed size");b=Number(l.getBigUint64(G+Y,!0)),Y+=8}if(E===4294967295){if(Y+8>V)throw new Error("zip64 extended information extra field does not include compressed size");E=Number(l.getBigUint64(G+Y,!0)),Y+=8}if(M===4294967295){if(Y+8>V)throw new Error("zip64 extended information extra field does not include relative header offset");M=Number(l.getBigUint64(G+Y,!0)),Y+=8}}let R=S.find(L=>L.id===28789&&L.length>=6&&a[L.offset]===1&&l.getInt32(L.offset+1,!0)===(0,yw.buf)(P));if(R&&(P=a.slice(R.offset+5,R.offset+R.length),F=!0),g===0){let L=b;if((m&1)!==0&&(L+=12),E!==L)throw new Error(`compressed/uncompressed size mismatch for stored file: ${E} != ${L}`)}let N=u.decode(P);N=N.replaceAll("\\","/");let U={versionMadeBy:p,versionNeededToExtract:d,generalPurposeBitFlag:m,compressionMethod:g,lastModFileTime:y,lastModFileDate:I,crc32:_,compressedSize:E,uncompressedSize:b,nameBytes:P,commentBytes:O,internalFileAttributes:x,externalFileAttributes:k,relativeOffsetOfLocalHeader:M,fileName:N};c.push(U)}return{commentBytes:i,entries:c,sizeEstimate:i.length+a.length*2}}async function xw(e,t,r){if(t.generalPurposeBitFlag&1)throw new Error("encrypted entries not supported");let n=await e(t.relativeOffsetOfLocalHeader,30,r),i=new DataView(n.buffer,n.byteOffset,n.byteLength),s=i.getUint32(0,!0);if(s!==67324752)throw new Error(`invalid local file header signature: 0x${s.toString(16)}`);let o=i.getUint16(26,!0),a=i.getUint16(28,!0);return t.relativeOffsetOfLocalHeader+n.length+o+a}async function Sw(e,t,r){return await e3(ZD(e),t,r)}var Bc=(e=>(e[e.STORE=0]="STORE",e[e.DEFLATE=8]="DEFLATE",e))(Bc||{});var ww=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),Ew=e=>{throw TypeError(e)},i3=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&Ew("Object expected");var n,i;r&&(n=t[ww("asyncDispose")]),n===void 0&&(n=t[ww("dispose")],r&&(i=n)),typeof n!="function"&&Ew("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},s3=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,l){return l=Error(c),l.name="SuppressedError",l.error=o,l.suppressed=a,l},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};function Iw(e){return async(t,r,n)=>{let i=await Pr(e,{throwIfMissing:!0,byteRange:{offset:t,length:r},strictByteRange:!0,signal:n.signal,progressListener:n.progressListener});return new Uint8Array(await i.response.arrayBuffer())}}function o3(e,t){let r=t.getUrl();return Sv(e,`zipMetadata:${r}`,{get:async(n,i)=>{var s=[];try{let c=i3(s,new He(i.progressListener,{message:`Reading ZIP central directory from ${r}`})),l=await t.stat(i);if(l?.totalSize===void 0)throw new Error(`Failed to determine ZIP file size: ${r}`);let u=await Sw(Iw(t),l.totalSize,i);return Zm(u.entries,f=>!f.fileName.endsWith("/")),u.entries.sort((f,h)=>St(f.fileName,h.fileName)),{data:u,size:u.sizeEstimate}}catch(c){var o=c,a=!0}finally{s3(s,o,a)}}})}async function a3(e,t,r){let n=o3(e,t);try{return await n.get(void 0,r)}finally{n.dispose()}}function bw(e,t){let{entries:r}=e,n=er(r,t,(i,s)=>St(i,s.fileName));if(!(n<0))return r[n]}function c3(e,t){let{entries:r}=e,n=we(0,r.length,a=>r[a].fileName>=t),i=we(Math.min(r.length,n+1),r.length,a=>!r[a].fileName.startsWith(t)),s=[],o=[];for(let a=n;a<i;){let c=r[a],l=c.fileName.indexOf("/",t.length);if(l===-1)s.push({key:c.fileName}),++a;else{o.push(c.fileName.substring(0,l));let u=c.fileName.substring(0,l+1);a=we(a+1,i,f=>!r[f].fileName.startsWith(u))}}return{entries:s,directories:o}}var zc=class{constructor(t,r){this.chunkManager=t,this.base=r}metadata;async getMetadata(t){let{metadata:r}=this;return r===void 0&&(r=this.metadata=await a3(this.chunkManager,this.base,t)),r}getUrl(t){return this.base.getUrl()+`|zip:${Ze(t)}`}async stat(t,r){let n=bw(await this.getMetadata(r),t);if(n!==void 0)return{totalSize:n.uncompressedSize}}async read(t,r){let n=bw(await this.getMetadata(r),t);if(n===void 0)return;let{fileDataStart:i}=n;i===void 0&&(i=n.fileDataStart=await xw(Iw(this.base),n,r));let s=new tt(this.base,{offset:i,length:n.compressedSize});switch(n.compressionMethod){case Bc.STORE:break;case Bc.DEFLATE:s=new Gr(s,"deflate-raw");break;default:throw new Error(`Unsupported compression method: ${n.compressionMethod}`)}return s.read(r)}async list(t,r){let n=await this.getMetadata(r);return c3(n,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function l3(e){return{scheme:"zip",description:"ZIP archive",getKvStore(t,r){return Rr(t),{store:new zc(e.chunkManager,new ke(r.store,r.path)),path:decodeURIComponent(t.suffix??"")}}}}wt.registerKvStoreAdapterProvider(l3);var _w=new to(self,!1);_w.sendReady();globalThis.rpc=_w;
/*! Bundled license information:

crc-32/crc32c.js:
  (*! crc32.js (C) 2014-present SheetJS -- http://sheetjs.com *)

crc-32/crc32.js:
  (*! crc32.js (C) 2014-present SheetJS -- http://sheetjs.com *)

lodash-es/lodash.js:
  (**
   * @license
   * Lodash (Custom Build) <https://lodash.com/>
   * Build: `lodash modularize exports="es" --repo lodash/lodash#4.18.1 -o ./`
   * Copyright OpenJS Foundation and other contributors <https://openjsf.org/>
   * Released under MIT license <https://lodash.com/license>
   * Based on Underscore.js 1.8.3 <http://underscorejs.org/LICENSE>
   * Copyright Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
   *)

neuroglancer/lib/util/disposable.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/signal.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/trackable_value.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/abort.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/progress_listener.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/worker_rpc.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/shared_watchable_value.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/chunk_manager/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/linked_list.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/bigint.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/array.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/geom.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/json.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/memoize.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/pairing_heap.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/chunk_manager/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/credentials_provider/index.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/credentials_provider/shared_common.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/credentials_provider/shared_counterpart.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/string.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/index.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/url.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/auto_detect.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/context.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/register.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/shared_common.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/backend.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/render_layer_common.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/render_layer_backend.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/matrix.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/si_units.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/vector.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/coordinate_transform.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/trackable.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/trackable_enum.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/navigation_state.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/animation_frame_debounce.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/framerate.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/trackable_screenshot_mode.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use viewer file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/webgl/context.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/display_context.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/projection_parameters.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/render_coordinate_transform.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/chunk_layout.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/data_type.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/erf.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/velocity_estimation.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/visibility_priority/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/perspective_view/base.js:
  (**
   * @license
   * Copyright 2018 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/perspective_view/backend.js:
  (**
   * @license
   * Copyright 2018 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/volume_rendering/base.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/volume_rendering/backend.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/annotation/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/segmentation_graph/segment_id.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/disjoint_sets.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/shared_disjoint_sets.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/gpu_hash/hash_function.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/random.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/gpu_hash/hash_table.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/uint64_map.js:
  (**
   * @license
   * This work is a derivative of the Google Neuroglancer project,
   * Copyright 2016 Google Inc.
   * The Derivative Work is covered by
   * Copyright 2019 Howard Hughes Medical Institute
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/uint64_set.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/segmentation_display_state/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/segmentation_display_state/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/annotation/backend.js:
  (**
   * @license
   * Copyright 2018 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/http_request.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/credentials_provider/http_request.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/boss/api.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/boss/base.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/mesh/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/zorder.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/mesh/multiscale.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/mesh/triangle_strips.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/endian.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/mesh/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/index.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/encode_compressed_segmentation_request.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/request.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/postprocess.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/gzip.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/numpy_dtype.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/npy.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/bossNpz.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/decode_jpeg_request.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/jpeg.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/volume/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/boss/backend.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/float32_to_string.js:
  (**
   * @license
   * Copyright 2018 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/hex.js:
  (**
   * @license
   * Copyright 2017 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/color.js:
  (**
   * @license
   * Copyright 2018 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/float.js:
  (**
   * @license
   * Copyright 2021 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/lerp.js:
  (**
   * @license
   * Copyright 2021 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/annotation/index.js:
  (**
   * @license
   * Copyright 2018 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/credentials_provider/oauth2.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/brainmaps/api.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/brainmaps/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/skeleton/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/skeleton/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/compressed_segmentation.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/raw.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/brainmaps/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/decode_png_request.js:
  (**
   * @license
   * Copyright 2022 William Silversmith
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/deepzoom/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc., 2023 Gergely Csucs
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/deepzoom/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc., 2023 Gergely Csucs
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/dvid/api.js:
  (**
   * @license
   * This work is a derivative of the Google Neuroglancer project,
   * Copyright 2016 Google Inc.
   * The Derivative Work is covered by
   * Copyright 2019 Howard Hughes Medical Institute
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/dvid/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/skeleton/decode_swc_skeleton.js:
  (**
   * @license
   * This work is a derivative of the Google Neuroglancer project,
   * Copyright 2016 Google Inc.
   * The Derivative Work is covered by
   * Copyright 2020 Howard Hughes Medical Institute
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/dvid/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/byte_range/file_handle.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/http/read.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/http/common.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/graphene/base.js:
  (**
   * @license
   * Copyright 2019 The Neuroglancer Authors
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/precomputed/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/object_id.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/chunk_manager/generic_file_source.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/gzip/file_handle.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/hash.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/precomputed/sharded.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/mesh/draco/index.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/skeleton/decode_precomputed_skeleton.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/decode_compresso_request.js:
  (**
   * @license
   * Copyright 2021 William Silversmith
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/compresso.js:
  (**
   * @license
   * Copyright 2021 William Silvermsith.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/decode_jxl_request.js:
  (**
   * @license
   * Copyright 2024 William Silversmith
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/jxl.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/backend_chunk_decoders/png.js:
  (**
   * @license
   * Copyright 2022 William Silvermsith.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/precomputed/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/graphene/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/decode_blosc_request.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/decode_zstd_request.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/n5/base.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/n5/backend.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/nifti/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/sliceview/volume/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/nifti/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/obj_mesh_request.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/single_mesh/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/single_mesh/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/obj/backend.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/render/base.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/render/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/async_computation/vtk_mesh_request.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/vtk/backend.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/index.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/blosc/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/zstd/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/bytes/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/crc32c/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/base.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/gzip/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/metadata/parse_util.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/resolve.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/metadata/index.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/metadata/parse.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/sharding_indexed/resolve.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/sharding_indexed/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/codec/transpose/decode.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/datasource/zarr/backend.js:
  (**
   * @license
   * Copyright 2020 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/byte_range/index.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/byte_range/register.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/gcs/index.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/gcs/register.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/gzip/index.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/gzip/register.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/proxy.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/http/backend.js:
  (**
   * @license
   * Copyright 2023 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/http/register_backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/list.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/crockford_base32.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)
  (**
   * @license
   * MIT License
   *
   * Copyright (c) 2016-2021 Linus Unnebäck
   *
   * Permission is hereby granted, free of charge, to any person obtaining a copy
   * of this software and associated documentation files (the "Software"), to deal
   * in the Software without restriction, including without limitation the rights
   * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   * copies of the Software, and to permit persons to whom the Software is
   * furnished to do so, subject to the following conditions:
   *
   * The above copyright notice and this permission notice shall be included in all
   * copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   * SOFTWARE.
   *)

neuroglancer/lib/kvstore/icechunk/decode_utils.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/manifest.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/ref.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/snapshot.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/metadata_cache.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/read.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/url.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/complete_url.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/icechunk/register_backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/middleauth/common.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/middleauth/register_backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ngauth/register.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/util/leb128.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/decode_utils.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/key.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/indirect_data_reference.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/btree.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/version_tree.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/manifest.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/metadata_cache.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/list.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/read.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/version_specifier.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/read_version.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/url.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/list_versions.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/complete_url.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/ocdbt/register_backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/s3/list.js:
  (**
   * @license
   * Copyright 2019 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/s3/common.js:
  (**
   * @license
   * Copyright 2024 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/s3/backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/s3/register_backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/zip/metadata.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)
  (**
   * Derived from https://github.com/greggman/unzipit/blob/4d94c9b77f7815062ff4460311e8b3ce4f7d5deb/src/unzipit.js
   *
   * Includes only parsing of raw entries.
   *
   * @license
   *
   * The MIT License (MIT)
   *
   * Copyright (c) 2014 Josh Wolfe
   *
   * Permission is hereby granted, free of charge, to any person obtaining a copy
   * of this software and associated documentation files (the "Software"), to deal
   * in the Software without restriction, including without limitation the rights
   * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   * copies of the Software, and to permit persons to whom the Software is
   * furnished to do so, subject to the following conditions:
   *
   * The above copyright notice and this permission notice shall be included in
   * all copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   * SOFTWARE.
   *
   * MIT License
   *
   * Copyright (c) 2019 Gregg Tavares
   *
   * Permission is hereby granted, free of charge, to any person obtaining a copy
   * of this software and associated documentation files (the "Software"), to deal
   * in the Software without restriction, including without limitation the rights
   * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   * copies of the Software, and to permit persons to whom the Software is
   * furnished to do so, subject to the following conditions:
   *
   * The above copyright notice and this permission notice shall be included in
   * all copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   * SOFTWARE.
   *)

neuroglancer/lib/kvstore/zip/backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/kvstore/zip/register_backend.js:
  (**
   * @license
   * Copyright 2025 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)

neuroglancer/lib/worker_rpc_context.js:
  (**
   * @license
   * Copyright 2016 Google Inc.
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   *)
*/
