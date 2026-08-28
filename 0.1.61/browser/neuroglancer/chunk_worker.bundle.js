var kw=Object.create;var Gc=Object.defineProperty;var Nw=Object.getOwnPropertyDescriptor;var Dw=Object.getOwnPropertyNames;var Pw=Object.getPrototypeOf,Rw=Object.prototype.hasOwnProperty;var q=(e,t)=>()=>(t||e((t={exports:{}}).exports,t),t.exports),Pi=(e,t)=>{for(var r in t)Gc(e,r,{get:t[r],enumerable:!0})},Ow=(e,t,r,n)=>{if(t&&typeof t=="object"||typeof t=="function")for(let i of Dw(t))!Rw.call(e,i)&&i!==r&&Gc(e,i,{get:()=>t[i],enumerable:!(n=Nw(t,i))||n.enumerable});return e};var Ri=(e,t,r)=>(r=e!=null?kw(Pw(e)):{},Ow(t||!e||!e.__esModule?Gc(r,"default",{value:e,enumerable:!0}):r,e));var At=q((jc,Kh)=>{"use strict";var Oi=function(e){return e&&e.Math===Math&&e};Kh.exports=Oi(typeof globalThis=="object"&&globalThis)||Oi(typeof window=="object"&&window)||Oi(typeof self=="object"&&self)||Oi(typeof global=="object"&&global)||Oi(typeof jc=="object"&&jc)||function(){return this}()||Function("return this")()});var Jh=q((p3,Yh)=>{"use strict";var Uw=At();Yh.exports=Uw});var Qr=q((d3,Hh)=>{"use strict";Hh.exports=function(e){try{return!!e()}catch{return!0}}});var qc=q((m3,Wh)=>{"use strict";var Lw=Qr();Wh.exports=!Lw(function(){var e=function(){}.bind();return typeof e!="function"||e.hasOwnProperty("prototype")})});var bn=q((g3,Qh)=>{"use strict";var Xh=qc(),Zh=Function.prototype,Kc=Zh.call,Fw=Xh&&Zh.bind.bind(Kc,Kc);Qh.exports=Xh?Fw:function(e){return function(){return Kc.apply(e,arguments)}}});var Yc=q((y3,ep)=>{"use strict";ep.exports=function(e){return e==null}});var Jc=q((v3,tp)=>{"use strict";var Bw=Yc(),zw=TypeError;tp.exports=function(e){if(Bw(e))throw new zw("Can't call method on "+e);return e}});var np=q((x3,rp)=>{"use strict";var $w=Jc(),Vw=Object;rp.exports=function(e){return Vw($w(e))}});var $s=q((S3,ip)=>{"use strict";var Gw=bn(),jw=np(),qw=Gw({}.hasOwnProperty);ip.exports=Object.hasOwn||function(t,r){return qw(jw(t),r)}});var op=q((w3,sp)=>{"use strict";sp.exports=!1});var up=q((E3,cp)=>{"use strict";var ap=At(),Kw=Object.defineProperty;cp.exports=function(e,t){try{Kw(ap,e,{value:t,configurable:!0,writable:!0})}catch{ap[e]=t}return t}});var pp=q((b3,hp)=>{"use strict";var Yw=op(),Jw=At(),Hw=up(),lp="__core-js_shared__",fp=hp.exports=Jw[lp]||Hw(lp,{});(fp.versions||(fp.versions=[])).push({version:"3.50.0",mode:Yw?"pure":"global",copyright:"\xA9 2013\u20132025 Denis Pushkarev (zloirock.ru), 2025\u20132026 CoreJS Company (core-js.io). All rights reserved.",license:"https://github.com/zloirock/core-js/blob/v3.50.0/LICENSE",source:"https://github.com/zloirock/core-js"})});var gp=q((I3,mp)=>{"use strict";var dp=pp(),Ww=Object.create||Object;mp.exports=function(e,t){return dp[e]||(dp[e]=t||Ww(null))}});var vp=q((_3,yp)=>{"use strict";var Xw=bn(),Zw=0,Qw=Math.random(),eE=Xw(1.1.toString);yp.exports=function(e){return"Symbol("+(e===void 0?"":e)+")_"+eE(++Zw+Qw,36)}});var Ep=q((C3,wp)=>{"use strict";var tE=At(),xp=tE.navigator,Sp=xp&&xp.userAgent;wp.exports=Sp?String(Sp):""});var Tp=q((A3,Mp)=>{"use strict";var Ap=At(),Hc=Ep(),bp=Ap.process,Ip=Ap.Deno,_p=bp&&bp.versions||Ip&&Ip.version,Cp=_p&&_p.v8,Vt,Vs;Cp&&(Vt=Cp.split("."),Vs=Vt[0]>0&&Vt[0]<4?1:+(Vt[0]+Vt[1]));!Vs&&Hc&&(Vt=Hc.match(/Edge\/(\d+)/),(!Vt||Vt[1]>=74)&&(Vt=Hc.match(/Chrome\/(\d+)/),Vt&&(Vs=+Vt[1])));Mp.exports=Vs});var Wc=q((M3,Np)=>{"use strict";var kp=Tp(),rE=Qr(),nE=At(),iE=nE.String;Np.exports=!!Object.getOwnPropertySymbols&&!rE(function(){var e=Symbol("symbol detection");return!iE(e)||!(Object(e)instanceof Symbol)||!Symbol.sham&&kp&&kp<41})});var Xc=q((T3,Dp)=>{"use strict";var sE=Wc();Dp.exports=sE&&!Symbol.sham&&typeof Symbol.iterator=="symbol"});var Qc=q((k3,Rp)=>{"use strict";var oE=At(),aE=gp(),Pp=$s(),cE=vp(),uE=Wc(),lE=Xc(),In=oE.Symbol,Zc=aE("wks"),fE=lE?In.for||In:In&&In.withoutSetter||cE;Rp.exports=function(e){return Pp(Zc,e)||(Zc[e]=uE&&Pp(In,e)?In[e]:fE("Symbol."+e)),Zc[e]}});var Gs=q(Op=>{"use strict";var hE=Qc();Op.f=hE});var Ui=q((D3,Up)=>{"use strict";var pE=Qr();Up.exports=!pE(function(){return Object.defineProperty({},1,{get:function(){return 7}})[1]!==7})});var _n=q((P3,Lp)=>{"use strict";var eu=typeof document=="object"&&document.all;Lp.exports=typeof eu>"u"&&eu!==void 0?function(e){return typeof e=="function"||e===eu}:function(e){return typeof e=="function"}});var Li=q((R3,Fp)=>{"use strict";var dE=_n();Fp.exports=function(e){return typeof e=="object"?e!==null:dE(e)}});var $p=q((O3,zp)=>{"use strict";var mE=At(),Bp=Li(),tu=mE.document,gE=Bp(tu)&&Bp(tu.createElement);zp.exports=function(e){return gE?tu.createElement(e):{}}});var ru=q((U3,Vp)=>{"use strict";var yE=Ui(),vE=Qr(),xE=$p();Vp.exports=!yE&&!vE(function(){return Object.defineProperty(xE("div"),"a",{get:function(){return 7}}).a!==7})});var jp=q((L3,Gp)=>{"use strict";var SE=Ui(),wE=Qr();Gp.exports=SE&&wE(function(){return Object.defineProperty(function(){},"prototype",{value:42,writable:!1}).prototype!==42})});var Kp=q((F3,qp)=>{"use strict";var EE=Li(),bE=String,IE=TypeError;qp.exports=function(e){if(EE(e))return e;throw new IE(bE(e)+" is not an object")}});var qs=q((B3,Yp)=>{"use strict";var _E=qc(),js=Function.prototype.call;Yp.exports=_E?js.bind(js):function(){return js.apply(js,arguments)}});var Hp=q((z3,Jp)=>{"use strict";var nu=At(),CE=_n(),AE=function(e){return CE(e)?e:void 0};Jp.exports=function(e,t){return arguments.length<2?AE(nu[e]):nu[e]&&nu[e][t]}});var Xp=q(($3,Wp)=>{"use strict";var ME=bn();Wp.exports=ME({}.isPrototypeOf)});var iu=q((V3,Zp)=>{"use strict";var TE=Hp(),kE=_n(),NE=Xp(),DE=Xc(),PE=Object;Zp.exports=DE?function(e){return typeof e=="symbol"}:function(e){var t=TE("Symbol");return kE(t)&&NE(t.prototype,PE(e))}});var ed=q((G3,Qp)=>{"use strict";var RE=String;Qp.exports=function(e){try{return RE(e)}catch{return"Object"}}});var rd=q((j3,td)=>{"use strict";var OE=_n(),UE=ed(),LE=TypeError;td.exports=function(e){if(OE(e))return e;throw new LE(UE(e)+" is not a function")}});var id=q((q3,nd)=>{"use strict";var FE=rd(),BE=Yc();nd.exports=function(e,t){var r=e[t];return BE(r)?void 0:FE(r)}});var od=q((K3,sd)=>{"use strict";var su=qs(),ou=_n(),au=Li(),zE=TypeError;sd.exports=function(e,t){var r,n;if(t==="string"&&ou(r=e.toString)&&!au(n=su(r,e))||ou(r=e.valueOf)&&!au(n=su(r,e))||t!=="string"&&ou(r=e.toString)&&!au(n=su(r,e)))return n;throw new zE("Can't convert object to primitive value")}});var ld=q((Y3,ud)=>{"use strict";var $E=qs(),ad=Li(),cd=iu(),VE=id(),GE=od(),jE=Qc(),qE=TypeError,KE=jE("toPrimitive");ud.exports=function(e,t){if(!ad(e)||cd(e))return e;var r=VE(e,KE),n;if(r){if(t===void 0&&(t="default"),n=$E(r,e,t),!ad(n)||cd(n))return n;throw new qE("Can't convert object to primitive value")}return t===void 0&&(t="number"),GE(e,t)}});var cu=q((J3,fd)=>{"use strict";var YE=ld(),JE=iu();fd.exports=function(e){var t=YE(e,"string");return JE(t)?t:t+""}});var Ys=q(pd=>{"use strict";var HE=Ui(),WE=ru(),XE=jp(),Ks=Kp(),hd=cu(),ZE=TypeError,uu=Object.defineProperty,QE=Object.getOwnPropertyDescriptor,lu="enumerable",fu="configurable",hu="writable";pd.f=HE?XE?function(t,r,n){if(Ks(t),r=hd(r),Ks(n),typeof t=="function"&&r==="prototype"&&"value"in n&&hu in n&&!n[hu]){var i=QE(t,r);i&&i[hu]&&(t[r]=n.value,n={configurable:fu in n?n[fu]:i[fu],enumerable:lu in n?n[lu]:i[lu],writable:!1})}return uu(t,r,n)}:uu:function(t,r,n){if(Ks(t),r=hd(r),Ks(n),WE)try{return uu(t,r,n)}catch{}if("get"in n||"set"in n)throw new ZE("Accessors not supported");return"value"in n&&(t[r]=n.value),t}});var pu=q((W3,md)=>{"use strict";var dd=Jh(),eb=$s(),tb=Gs(),rb=Ys().f;md.exports=function(e){var t=dd.Symbol||(dd.Symbol={});eb(t,e)||rb(t,e,{value:tb.f(e)})}});var xd=q(vd=>{"use strict";var gd={}.propertyIsEnumerable,yd=Object.getOwnPropertyDescriptor,nb=yd&&!gd.call({1:2},1);vd.f=nb?function(t){var r=yd(this,t);return!!r&&r.enumerable}:gd});var wd=q((Z3,Sd)=>{"use strict";Sd.exports=function(e,t){return{enumerable:!(e&1),configurable:!(e&2),writable:!(e&4),value:t}}});var Id=q((Q3,bd)=>{"use strict";var Ed=bn(),ib=Ed({}.toString),sb=Ed("".slice);bd.exports=function(e){return sb(ib(e),8,-1)}});var Cd=q((eP,_d)=>{"use strict";var ob=bn(),ab=Qr(),cb=Id(),du=Object,ub=ob("".split);_d.exports=ab(function(){return!du("z").propertyIsEnumerable(0)})?function(e){return cb(e)==="String"?ub(e,""):du(e)}:du});var Md=q((tP,Ad)=>{"use strict";var lb=Cd(),fb=Jc();Ad.exports=function(e){return lb(fb(e))}});var mu=q(kd=>{"use strict";var hb=Ui(),pb=qs(),db=xd(),mb=wd(),gb=Md(),yb=cu(),vb=$s(),xb=ru(),Td=Object.getOwnPropertyDescriptor;kd.f=hb?Td:function(t,r){if(t=gb(t),r=yb(r),xb)try{return Td(t,r)}catch{}if(vb(t,r))return mb(!pb(db.f,t,r),t[r])}});var yu=q(()=>{"use strict";var Sb=At(),wb=pu(),Eb=Ys().f,bb=mu().f,gu=Sb.Symbol;wb("dispose");gu&&(Fi=bb(gu,"dispose"),Fi.enumerable&&Fi.configurable&&Fi.writable&&Eb(gu,"dispose",{value:Fi.value,enumerable:!1,configurable:!1,writable:!1}));var Fi});var Dd=q((sP,Nd)=>{"use strict";yu();var Ib=Gs();Nd.exports=Ib.f("dispose")});var Rd=q((oP,Pd)=>{"use strict";var _b=Dd();Pd.exports=_b});var Od=q(()=>{"use strict";yu()});var Ld=q((uP,Ud)=>{"use strict";var Cb=Rd();Od();Ud.exports=Cb});var xu=q(()=>{"use strict";var Ab=At(),Mb=pu(),Tb=Ys().f,kb=mu().f,vu=Ab.Symbol;Mb("asyncDispose");vu&&(Bi=kb(vu,"asyncDispose"),Bi.enumerable&&Bi.configurable&&Bi.writable&&Tb(vu,"asyncDispose",{value:Bi.value,enumerable:!1,configurable:!1,writable:!1}));var Bi});var Bd=q((hP,Fd)=>{"use strict";xu();var Nb=Gs();Fd.exports=Nb.f("asyncDispose")});var $d=q((pP,zd)=>{"use strict";var Db=Bd();zd.exports=Db});var Vd=q(()=>{"use strict";xu()});var jd=q((gP,Gd)=>{"use strict";var Pb=$d();Vd();Gd.exports=Pb});var kx=q(H=>{"use strict";H.deflate=yx;H.deflateSync=vs;H.inflate=Rf;H.inflateSync=vi;H.gzip=qa;H.compress=qa;H.gzipSync=La;H.compressSync=La;H.gunzip=Sx;H.gunzipSync=Ba;H.zlib=bT;H.zlibSync=bf;H.unzlib=Ex;H.unzlibSync=$a;H.gzip=qa;H.compress=qa;H.decompress=_T;H.decompressSync=CT;H.strToU8=Yr;H.strFromU8=Uf;H.zip=DT;H.zipSync=PT;H.unzip=LT;H.unzipSync=FT;var Jv={},Xv={};Xv.default=function(e,t,r,n,i){var s=new Worker(Jv[t]||(Jv[t]=URL.createObjectURL(new Blob([e+';addEventListener("error",function(e){e=e.error;postMessage({$e$:[e.message,e.code,e.stack]})})'],{type:"text/javascript"}))));return s.onmessage=function(o){var a=o.data,c=a.$e$;if(c){var u=new Error(c[0]);u.code=c[1],u.stack=c[2],i(u,null)}else i(null,a)},s.postMessage(r,n),s};var ee=Uint8Array,nt=Uint16Array,ms=Int32Array,li=new ee([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0]),fi=new ee([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,0,0]),hs=new ee([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]),Zv=function(e,t){for(var r=new nt(31),n=0;n<31;++n)r[n]=t+=1<<e[n-1];for(var i=new ms(r[30]),n=1;n<30;++n)for(var s=r[n];s<r[n+1];++s)i[s]=s-r[n]<<5|n;return{b:r,r:i}},Qv=Zv(li,2),Cf=Qv.b,Ra=Qv.r;Cf[28]=258,Ra[258]=28;var ex=Zv(fi,0),tx=ex.b,xf=ex.r,ps=new nt(32768);for(pe=0;pe<32768;++pe)Ir=(pe&43690)>>1|(pe&21845)<<1,Ir=(Ir&52428)>>2|(Ir&13107)<<2,Ir=(Ir&61680)>>4|(Ir&3855)<<4,ps[pe]=((Ir&65280)>>8|(Ir&255)<<8)>>1;var Ir,pe,Rt=function(e,t,r){for(var n=e.length,i=0,s=new nt(t);i<n;++i)e[i]&&++s[e[i]-1];var o=new nt(t);for(i=1;i<t;++i)o[i]=o[i-1]+s[i-1]<<1;var a;if(r){a=new nt(1<<t);var c=15-t;for(i=0;i<n;++i)if(e[i])for(var u=i<<4|e[i],l=t-e[i],f=o[e[i]-1]++<<l,h=f|(1<<l)-1;f<=h;++f)a[ps[f]>>c]=u}else for(a=new nt(n),i=0;i<n;++i)e[i]&&(a[i]=ps[o[e[i]-1]++]>>15-e[i]);return a},_r=new ee(288);for(pe=0;pe<144;++pe)_r[pe]=8;var pe;for(pe=144;pe<256;++pe)_r[pe]=9;var pe;for(pe=256;pe<280;++pe)_r[pe]=7;var pe;for(pe=280;pe<288;++pe)_r[pe]=8;var pe,ci=new ee(32);for(pe=0;pe<32;++pe)ci[pe]=5;var pe,rx=Rt(_r,9,0),nx=Rt(_r,9,1),ix=Rt(ci,5,0),sx=Rt(ci,5,1),Na=function(e){for(var t=e[0],r=1;r<e.length;++r)e[r]>t&&(t=e[r]);return t},Pt=function(e,t,r){var n=t/8|0;return(e[n]|e[n+1]<<8)>>(t&7)&r},Da=function(e,t){var r=t/8|0;return(e[r]|e[r+1]<<8|e[r+2]<<16)>>(t&7)},hi=function(e){return(e+7)/8|0},Ot=function(e,t,r){return(t==null||t<0)&&(t=0),(r==null||r>e.length)&&(r=e.length),new ee(e.subarray(t,r))};H.FlateErrorCode={UnexpectedEOF:0,InvalidBlockType:1,InvalidLengthLiteral:2,InvalidDistance:3,StreamFinished:4,NoStreamHandler:5,InvalidHeader:6,NoCallback:7,InvalidUTF8:8,ExtraFieldTooLong:9,InvalidDate:10,FilenameTooLong:11,StreamFinishing:12,InvalidZipData:13,UnknownCompressionMethod:14};var ox=["unexpected EOF","invalid block type","invalid length/literal","invalid distance","stream finished","no stream handler",,"no callback","invalid UTF-8 data","extra field too long","date not in range 1980-2099","filename too long","stream finishing","invalid zip data"],K=function(e,t,r){var n=new Error(t||ox[e]);if(n.code=e,Error.captureStackTrace&&Error.captureStackTrace(n,K),!r)throw n;return n},gs=function(e,t,r,n){var i=e.length,s=n?n.length:0;if(!i||t.f&&!t.l)return r||new ee(0);var o=!r,a=o||t.i!=2,c=t.i;o&&(r=new ee(i*3));var u=function(vt){var dr=r.length;if(vt>dr){var Tr=new ee(Math.max(dr*2,vt));Tr.set(r),r=Tr}},l=t.f||0,f=t.p||0,h=t.b||0,p=t.l,d=t.d,m=t.m,g=t.n,y=i*8;do{if(!p){l=Pt(e,f,1);var I=Pt(e,f+1,3);if(f+=3,I)if(I==1)p=nx,d=sx,m=9,g=5;else if(I==2){var C=Pt(e,f,31)+257,v=Pt(e,f+10,15)+4,w=C+Pt(e,f+5,31)+1;f+=14;for(var x=new ee(w),T=new ee(19),M=0;M<v;++M)T[hs[M]]=Pt(e,f+M*3,7);f+=v*3;for(var P=Na(T),F=(1<<P)-1,S=Rt(T,P,1),M=0;M<w;){var O=S[Pt(e,f,F)];f+=O&15;var _=O>>4;if(_<16)x[M++]=_;else{var R=0,N=0;for(_==16?(N=3+Pt(e,f,3),f+=2,R=x[M-1]):_==17?(N=3+Pt(e,f,7),f+=3):_==18&&(N=11+Pt(e,f,127),f+=7);N--;)x[M++]=R}}var U=x.subarray(0,C),L=x.subarray(C);m=Na(U),g=Na(L),p=Rt(U,m,1),d=Rt(L,g,1)}else K(1);else{var _=hi(f)+4,E=e[_-4]|e[_-3]<<8,b=_+E;if(b>i){c&&K(0);break}a&&u(h+E),r.set(e.subarray(_,b),h),t.b=h+=E,t.p=f=b*8,t.f=l;continue}if(f>y){c&&K(0);break}}a&&u(h+131072);for(var j=(1<<m)-1,V=(1<<g)-1,Y=f;;Y=f){var R=p[Da(e,f)&j],J=R>>4;if(f+=R&15,f>y){c&&K(0);break}if(R||K(2),J<256)r[h++]=J;else if(J==256){Y=f,p=null;break}else{var de=J-254;if(J>264){var M=J-257,ie=li[M];de=Pt(e,f,(1<<ie)-1)+Cf[M],f+=ie}var Ce=d[Da(e,f)&V],ze=Ce>>4;Ce||K(3),f+=Ce&15;var L=tx[ze];if(ze>3){var ie=fi[ze];L+=Da(e,f)&(1<<ie)-1,f+=ie}if(f>y){c&&K(0);break}a&&u(h+131072);var at=h+de;if(h<L){var gt=s-L,yt=Math.min(L,at);for(gt+h<0&&K(3);h<yt;++h)r[h]=n[gt+h]}for(;h<at;++h)r[h]=r[h-L]}}t.l=p,t.p=Y,t.b=h,t.f=l,p&&(l=1,t.m=m,t.d=d,t.n=g)}while(!l);return h!=r.length&&o?Ot(r,0,h):r.subarray(0,h)},ar=function(e,t,r){r<<=t&7;var n=t/8|0;e[n]|=r,e[n+1]|=r>>8},oi=function(e,t,r){r<<=t&7;var n=t/8|0;e[n]|=r,e[n+1]|=r>>8,e[n+2]|=r>>16},Pa=function(e,t){for(var r=[],n=0;n<e.length;++n)e[n]&&r.push({s:n,f:e[n]});var i=r.length,s=r.slice();if(!i)return{t:cr,l:0};if(i==1){var o=new ee(r[0].s+1);return o[r[0].s]=1,{t:o,l:1}}r.sort(function(b,C){return b.f-C.f}),r.push({s:-1,f:25001});var a=r[0],c=r[1],u=0,l=1,f=2;for(r[0]={s:-1,f:a.f+c.f,l:a,r:c};l!=i-1;)a=r[r[u].f<r[f].f?u++:f++],c=r[u!=l&&r[u].f<r[f].f?u++:f++],r[l++]={s:-1,f:a.f+c.f,l:a,r:c};for(var h=s[0].s,n=1;n<i;++n)s[n].s>h&&(h=s[n].s);var p=new nt(h+1),d=Oa(r[l-1],p,0);if(d>t){var n=0,m=0,g=d-t,y=1<<g;for(s.sort(function(C,v){return p[v.s]-p[C.s]||C.f-v.f});n<i;++n){var I=s[n].s;if(p[I]>t)m+=y-(1<<d-p[I]),p[I]=t;else break}for(m>>=g;m>0;){var _=s[n].s;p[_]<t?m-=1<<t-p[_]++-1:++n}for(;n>=0&&m;--n){var E=s[n].s;p[E]==t&&(--p[E],++m)}d=t}return{t:new ee(p),l:d}},Oa=function(e,t,r){return e.s==-1?Math.max(Oa(e.l,t,r+1),Oa(e.r,t,r+1)):t[e.s]=r},Sf=function(e){for(var t=e.length;t&&!e[--t];);for(var r=new nt(++t),n=0,i=e[0],s=1,o=function(c){r[n++]=c},a=1;a<=t;++a)if(e[a]==i&&a!=t)++s;else{if(!i&&s>2){for(;s>138;s-=138)o(32754);s>2&&(o(s>10?s-11<<5|28690:s-3<<5|12305),s=0)}else if(s>3){for(o(i),--s;s>6;s-=6)o(8304);s>2&&(o(s-3<<5|8208),s=0)}for(;s--;)o(i);s=1,i=e[a]}return{c:r.subarray(0,n),n:t}},ai=function(e,t){for(var r=0,n=0;n<t.length;++n)r+=e[n]*t[n];return r},Ga=function(e,t,r){var n=r.length,i=hi(t+2);e[i]=n&255,e[i+1]=n>>8,e[i+2]=e[i]^255,e[i+3]=e[i+1]^255;for(var s=0;s<n;++s)e[i+s+4]=r[s];return(i+4+n)*8},wf=function(e,t,r,n,i,s,o,a,c,u,l){ar(t,l++,r),++i[256];for(var f=Pa(i,15),h=f.t,p=f.l,d=Pa(s,15),m=d.t,g=d.l,y=Sf(h),I=y.c,_=y.n,E=Sf(m),b=E.c,C=E.n,v=new nt(19),w=0;w<I.length;++w)++v[I[w]&31];for(var w=0;w<b.length;++w)++v[b[w]&31];for(var x=Pa(v,7),T=x.t,M=x.l,P=19;P>4&&!T[hs[P-1]];--P);var F=u+5<<3,S=ai(i,_r)+ai(s,ci)+o,O=ai(i,h)+ai(s,m)+o+14+3*P+ai(v,T)+2*v[16]+3*v[17]+7*v[18];if(c>=0&&F<=S&&F<=O)return Ga(t,l,e.subarray(c,c+u));var R,N,U,L;if(ar(t,l,1+(O<S)),l+=2,O<S){R=Rt(h,p,0),N=h,U=Rt(m,g,0),L=m;var j=Rt(T,M,0);ar(t,l,_-257),ar(t,l+5,C-1),ar(t,l+10,P-4),l+=14;for(var w=0;w<P;++w)ar(t,l+3*w,T[hs[w]]);l+=3*P;for(var V=[I,b],Y=0;Y<2;++Y)for(var J=V[Y],w=0;w<J.length;++w){var de=J[w]&31;ar(t,l,j[de]),l+=T[de],de>15&&(ar(t,l,J[w]>>5&127),l+=J[w]>>12)}}else R=rx,N=_r,U=ix,L=ci;for(var w=0;w<a;++w){var ie=n[w];if(ie>255){var de=ie>>18&31;oi(t,l,R[de+257]),l+=N[de+257],de>7&&(ar(t,l,ie>>23&31),l+=li[de]);var Ce=ie&31;oi(t,l,U[Ce]),l+=L[Ce],Ce>3&&(oi(t,l,ie>>5&8191),l+=fi[Ce])}else oi(t,l,R[ie]),l+=N[ie]}return oi(t,l,R[256]),l+N[256]},ax=new ms([65540,131080,131088,131104,262176,1048704,1048832,2114560,2117632]),cr=new ee(0),cx=function(e,t,r,n,i,s){var o=s.z||e.length,a=new ee(n+o+5*(1+Math.ceil(o/7e3))+i),c=a.subarray(n,a.length-i),u=s.l,l=(s.r||0)&7;if(t){l&&(c[0]=s.r>>3);for(var f=ax[t-1],h=f>>13,p=f&8191,d=(1<<r)-1,m=s.p||new nt(32768),g=s.h||new nt(d+1),y=Math.ceil(r/3),I=2*y,_=function(Ti){return(e[Ti]^e[Ti+1]<<y^e[Ti+2]<<I)&d},E=new ms(25e3),b=new nt(288),C=new nt(32),v=0,w=0,x=s.i||0,T=0,M=s.w||0,P=0;x+2<o;++x){var F=_(x),S=x&32767,O=g[F];if(m[S]=O,g[F]=S,M<=x){var R=o-x;if((v>7e3||T>24576)&&(R>423||!u)){l=wf(e,c,0,E,b,C,w,T,P,x-P,l),T=v=w=0,P=x;for(var N=0;N<286;++N)b[N]=0;for(var N=0;N<30;++N)C[N]=0}var U=2,L=0,j=p,V=S-O&32767;if(R>2&&F==_(x-V))for(var Y=Math.min(h,R)-1,J=Math.min(32767,x),de=Math.min(258,R);V<=J&&--j&&S!=O;){if(e[x+U]==e[x+U-V]){for(var ie=0;ie<de&&e[x+ie]==e[x+ie-V];++ie);if(ie>U){if(U=ie,L=V,ie>Y)break;for(var Ce=Math.min(V,ie-2),ze=0,N=0;N<Ce;++N){var at=x-V+N&32767,gt=m[at],yt=at-gt&32767;yt>ze&&(ze=yt,O=at)}}}S=O,O=m[S],V+=S-O&32767}if(L){E[T++]=268435456|Ra[U]<<18|xf[L];var vt=Ra[U]&31,dr=xf[L]&31;w+=li[vt]+fi[dr],++b[257+vt],++C[dr],M=x+U,++v}else E[T++]=e[x],++b[e[x]]}}for(x=Math.max(x,M);x<o;++x)E[T++]=e[x],++b[e[x]];l=wf(e,c,u,E,b,C,w,T,P,x-P,l),u||(s.r=l&7|c[l/8|0]<<3,l-=7,s.h=g,s.p=m,s.i=x,s.w=M)}else{for(var x=s.w||0;x<o+u;x+=65535){var Tr=x+65535;Tr>=o&&(c[l/8|0]=u,Tr=o),l=Ga(c,l+1,e.subarray(x,Tr))}s.i=o}return Ot(a,0,n+hi(l)+i)},ux=function(){for(var e=new Int32Array(256),t=0;t<256;++t){for(var r=t,n=9;--n;)r=(r&1&&-306674912)^r>>>1;e[t]=r}return e}(),pi=function(){var e=-1;return{p:function(t){for(var r=e,n=0;n<t.length;++n)r=ux[r&255^t[n]]^r>>>8;e=r},d:function(){return~e}}},ja=function(){var e=1,t=0;return{p:function(r){for(var n=e,i=t,s=r.length|0,o=0;o!=s;){for(var a=Math.min(o+2655,s);o<a;++o)i+=n+=r[o];n=(n&65535)+15*(n>>16),i=(i&65535)+15*(i>>16)}e=n,t=i},d:function(){return e%=65521,t%=65521,(e&255)<<24|(e&65280)<<8|(t&255)<<8|t>>8}}},gn=function(e,t,r,n,i){if(!i&&(i={l:1},t.dictionary)){var s=t.dictionary.subarray(-32768),o=new ee(s.length+e.length);o.set(s),o.set(e,s.length),e=o,i.w=s.length}return cx(e,t.level==null?6:t.level,t.mem==null?i.l?Math.ceil(Math.max(8,Math.min(13,Math.log(e.length)))*1.5):20:12+t.mem,r,n,i)},ys=function(e,t){var r={};for(var n in e)r[n]=e[n];for(var n in t)r[n]=t[n];return r},Hv=function(e,t,r){for(var n=e(),i=e.toString(),s=i.slice(i.indexOf("[")+1,i.lastIndexOf("]")).replace(/\s+/g,"").split(","),o=0;o<n.length;++o){var a=n[o],c=s[o];if(typeof a=="function"){t+=";"+c+"=";var u=a.toString();if(a.prototype)if(u.indexOf("[native code]")!=-1){var l=u.indexOf(" ",8)+1;t+=u.slice(l,u.indexOf("(",l))}else{t+=u;for(var f in a.prototype)t+=";"+c+".prototype."+f+"="+a.prototype[f].toString()}else t+=u}else r[c]=a}return t},ka=[],wT=function(e){var t=[];for(var r in e)e[r].buffer&&t.push((e[r]=new e[r].constructor(e[r])).buffer);return t},lx=function(e,t,r,n){if(!ka[r]){for(var i="",s={},o=e.length-1,a=0;a<o;++a)i=Hv(e[a],i,s);ka[r]={c:Hv(e[o],i,s),e:s}}var c=ys({},ka[r].e);return(0,Xv.default)(ka[r].c+";onmessage=function(e){for(var k in e.data)self[k]=e.data[k];onmessage="+t.toString()+"}",r,c,wT(c),n)},di=function(){return[ee,nt,ms,li,fi,hs,Cf,tx,nx,sx,ps,ox,Rt,Na,Pt,Da,hi,Ot,K,gs,vi,Jr,Af]},mi=function(){return[ee,nt,ms,li,fi,hs,Ra,xf,rx,_r,ix,ci,ps,ax,cr,Rt,ar,oi,Pa,Oa,Sf,ai,Ga,wf,hi,Ot,cx,gn,vs,Jr]},fx=function(){return[Mf,kf,he,pi,ux]},hx=function(){return[Tf,mx]},px=function(){return[Nf,he,ja]},dx=function(){return[Df]},Jr=function(e){return postMessage(e,[e.buffer])},Af=function(e){return e&&{out:e.size&&new ee(e.size),dictionary:e.dictionary}},gi=function(e,t,r,n,i,s){var o=lx(r,n,i,function(a,c){o.terminate(),s(a,c)});return o.postMessage([e,t],t.consume?[e.buffer]:[]),function(){o.terminate()}},Lt=function(e){return e.ondata=function(t,r){return postMessage([t,r],[t.buffer])},function(t){t.data[0]?(e.push(t.data[0],t.data[1]),postMessage([t.data[0].length])):e.flush(t.data[1])}},yi=function(e,t,r,n,i,s,o){var a,c=lx(e,n,i,function(u,l){u?(c.terminate(),t.ondata.call(t,u)):Array.isArray(l)?l.length==1?(t.queuedSize-=l[0],t.ondrain&&t.ondrain(l[0])):(l[1]&&c.terminate(),t.ondata.call(t,u,l[0],l[1])):o(l)});c.postMessage(r),t.queuedSize=0,t.push=function(u,l){t.ondata||K(5),a&&t.ondata(K(4,0,1),null,!!l),t.queuedSize+=u.length,c.postMessage([u,a=l],u.buffer instanceof ArrayBuffer?[u.buffer]:[])},t.terminate=function(){c.terminate()},s&&(t.flush=function(u){c.postMessage([0,u])})},rt=function(e,t){return e[t]|e[t+1]<<8},Ie=function(e,t){return(e[t]|e[t+1]<<8|e[t+2]<<16|e[t+3]<<24)>>>0},vf=function(e,t){return Ie(e,t)+Ie(e,t+4)*4294967296},he=function(e,t,r){for(;r;++t)e[t]=r,r>>>=8},Mf=function(e,t){var r=t.filename;if(e[0]=31,e[1]=139,e[2]=8,e[8]=t.level<2?4:t.level==9?2:0,e[9]=3,t.mtime!=0&&he(e,4,Math.floor(new Date(t.mtime||Date.now())/1e3)),r){e[3]=8;for(var n=0;n<=r.length;++n)e[n+10]=r.charCodeAt(n)}},Tf=function(e){(e[0]!=31||e[1]!=139||e[2]!=8)&&K(6,"invalid gzip data");var t=e[3],r=10;t&4&&(r+=(e[10]|e[11]<<8)+2);for(var n=(t>>3&1)+(t>>4&1);n>0;n-=!e[r++]);return r+(t&2)},mx=function(e){var t=e.length;return(e[t-4]|e[t-3]<<8|e[t-2]<<16|e[t-1]<<24)>>>0},kf=function(e){return 10+(e.filename?e.filename.length+1:0)},Nf=function(e,t){var r=t.level,n=r==0?0:r<6?1:r==9?3:2;if(e[0]=120,e[1]=n<<6|(t.dictionary&&32),e[1]|=31-(e[0]<<8|e[1])%31,t.dictionary){var i=ja();i.p(t.dictionary),he(e,2,i.d())}},Df=function(e,t){return((e[0]&15)!=8||e[0]>>4>7||(e[0]<<8|e[1])%31)&&K(6,"invalid zlib data"),(e[1]>>5&1)==+!t&&K(6,"invalid zlib data: "+(e[1]&32?"need":"unexpected")+" dictionary"),(e[1]>>3&4)+2};function yn(e,t){return typeof e=="function"&&(t=e,e={}),this.ondata=t,e}var Ut=function(){function e(t,r){if(typeof t=="function"&&(r=t,t={}),this.ondata=r,this.o=t||{},this.s={l:0,i:32768,w:32768,z:32768},this.b=new ee(98304),this.o.dictionary){var n=this.o.dictionary.subarray(-32768);this.b.set(n,32768-n.length),this.s.i=32768-n.length}}return e.prototype.p=function(t,r){this.ondata(gn(t,this.o,0,0,this.s),r)},e.prototype.push=function(t,r){this.ondata||K(5),this.s.l&&K(4);var n=t.length+this.s.z;if(n>this.b.length){if(n>2*this.b.length-32768){var i=new ee(n&-32768);i.set(this.b.subarray(0,this.s.z)),this.b=i}var s=this.b.length-this.s.z;this.b.set(t.subarray(0,s),this.s.z),this.s.z=this.b.length,this.p(this.b,!1),this.b.set(this.b.subarray(-32768)),this.b.set(t.subarray(s),32768),this.s.z=t.length-s+32768,this.s.i=32766,this.s.w=32768}else this.b.set(t,this.s.z),this.s.z+=t.length;this.s.l=r&1,(this.s.z>this.s.w+8191||r)&&(this.p(this.b,r||!1),this.s.w=this.s.i,this.s.i-=2),r&&(this.s=this.o={},this.b=cr)},e.prototype.flush=function(t){if(this.ondata||K(5),this.s.l&&K(4),this.p(this.b,!1),this.s.w=this.s.i,this.s.i-=2,t){var r=new ee(6);r[0]=this.s.r>>3;var n=Ga(r,this.s.r,cr);this.s.r=0,this.ondata(r.subarray(0,n>>3),!1)}},e}();H.Deflate=Ut;var gx=function(){function e(t,r){yi([mi,function(){return[Lt,Ut]}],this,yn.call(this,t,r),function(n){var i=new Ut(n.data);onmessage=Lt(i)},6,1)}return e}();H.AsyncDeflate=gx;function yx(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[mi],function(n){return Jr(vs(n.data[0],n.data[1]))},0,r)}function vs(e,t){return gn(e,t||{},0,0)}var it=function(){function e(t,r){typeof t=="function"&&(r=t,t={}),this.ondata=r;var n=t&&t.dictionary&&t.dictionary.subarray(-32768);this.s={i:0,b:n?n.length:0},this.o=new ee(32768),this.p=new ee(0),n&&this.o.set(n)}return e.prototype.e=function(t){if(this.ondata||K(5),this.d&&K(4),!this.p.length)this.p=t;else if(t.length){var r=new ee(this.p.length+t.length);r.set(this.p),r.set(t,this.p.length),this.p=r}},e.prototype.c=function(t){this.s.i=+(this.d=t||!1);var r=this.s.b,n=gs(this.p,this.s,this.o);this.ondata(Ot(n,r,this.s.b),this.d),this.o=Ot(n,this.s.b-32768),this.s.b=this.o.length,this.p=Ot(this.p,this.s.p/8|0),this.s.p&=7},e.prototype.push=function(t,r){this.e(t),this.c(r)},e}();H.Inflate=it;var Pf=function(){function e(t,r){yi([di,function(){return[Lt,it]}],this,yn.call(this,t,r),function(n){var i=new it(n.data);onmessage=Lt(i)},7,0)}return e}();H.AsyncInflate=Pf;function Rf(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[di],function(n){return Jr(vi(n.data[0],Af(n.data[1])))},1,r)}function vi(e,t){return gs(e,{i:2},t&&t.out,t&&t.dictionary)}var Ua=function(){function e(t,r){this.c=pi(),this.l=0,this.v=1,Ut.call(this,t,r)}return e.prototype.push=function(t,r){this.c.p(t),this.l+=t.length,Ut.prototype.push.call(this,t,r)},e.prototype.p=function(t,r){var n=gn(t,this.o,this.v&&kf(this.o),r&&8,this.s);this.v&&(Mf(n,this.o),this.v=0),r&&(he(n,n.length-8,this.c.d()),he(n,n.length-4,this.l)),this.ondata(n,r)},e.prototype.flush=function(t){Ut.prototype.flush.call(this,t)},e}();H.Gzip=Ua;H.Compress=Ua;var vx=function(){function e(t,r){yi([mi,fx,function(){return[Lt,Ut,Ua]}],this,yn.call(this,t,r),function(n){var i=new Ua(n.data);onmessage=Lt(i)},8,1)}return e}();H.AsyncGzip=vx;H.AsyncCompress=vx;function qa(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[mi,fx,function(){return[La]}],function(n){return Jr(La(n.data[0],n.data[1]))},2,r)}function La(e,t){t||(t={});var r=pi(),n=e.length;r.p(e);var i=gn(e,t,kf(t),8),s=i.length;return Mf(i,t),he(i,s-8,r.d()),he(i,s-4,n),i}var Fa=function(){function e(t,r){this.v=1,this.r=0,it.call(this,t,r)}return e.prototype.push=function(t,r){if(it.prototype.e.call(this,t),this.r+=t.length,this.v){var n=this.p.subarray(this.v-1),i=n.length>3?Tf(n):4;if(i>n.length){if(!r)return}else this.v>1&&this.onmember&&this.onmember(this.r-n.length);this.p=n.subarray(i),this.v=0}it.prototype.c.call(this,0),this.s.f&&!this.s.l?(this.v=hi(this.s.p)+9,this.s={i:0},this.o=new ee(0),this.push(new ee(0),r)):r&&it.prototype.c.call(this,r)},e}();H.Gunzip=Fa;var xx=function(){function e(t,r){var n=this;yi([di,hx,function(){return[Lt,it,Fa]}],this,yn.call(this,t,r),function(i){var s=new Fa(i.data);s.onmember=function(o){return postMessage(o)},onmessage=Lt(s)},9,0,function(i){return n.onmember&&n.onmember(i)})}return e}();H.AsyncGunzip=xx;function Sx(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[di,hx,function(){return[Ba]}],function(n){return Jr(Ba(n.data[0],n.data[1]))},3,r)}function Ba(e,t){var r=Tf(e);return r+8>e.length&&K(6,"invalid gzip data"),gs(e.subarray(r,-8),{i:2},t&&t.out||new ee(mx(e)),t&&t.dictionary)}var Ef=function(){function e(t,r){this.c=ja(),this.v=1,Ut.call(this,t,r)}return e.prototype.push=function(t,r){this.c.p(t),Ut.prototype.push.call(this,t,r)},e.prototype.p=function(t,r){var n=gn(t,this.o,this.v&&(this.o.dictionary?6:2),r&&4,this.s);this.v&&(Nf(n,this.o),this.v=0),r&&he(n,n.length-4,this.c.d()),this.ondata(n,r)},e.prototype.flush=function(t){Ut.prototype.flush.call(this,t)},e}();H.Zlib=Ef;var ET=function(){function e(t,r){yi([mi,px,function(){return[Lt,Ut,Ef]}],this,yn.call(this,t,r),function(n){var i=new Ef(n.data);onmessage=Lt(i)},10,1)}return e}();H.AsyncZlib=ET;function bT(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[mi,px,function(){return[bf]}],function(n){return Jr(bf(n.data[0],n.data[1]))},4,r)}function bf(e,t){t||(t={});var r=ja();r.p(e);var n=gn(e,t,t.dictionary?6:2,4);return Nf(n,t),he(n,n.length-4,r.d()),n}var za=function(){function e(t,r){it.call(this,t,r),this.v=t&&t.dictionary?2:1}return e.prototype.push=function(t,r){if(it.prototype.e.call(this,t),this.v){if(this.p.length<6&&!r)return;this.p=this.p.subarray(Df(this.p,this.v-1)),this.v=0}r&&(this.p.length<4&&K(6,"invalid zlib data"),this.p=this.p.subarray(0,-4)),it.prototype.c.call(this,r)},e}();H.Unzlib=za;var wx=function(){function e(t,r){yi([di,dx,function(){return[Lt,it,za]}],this,yn.call(this,t,r),function(n){var i=new za(n.data);onmessage=Lt(i)},11,0)}return e}();H.AsyncUnzlib=wx;function Ex(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),gi(e,t,[di,dx,function(){return[$a]}],function(n){return Jr($a(n.data[0],Af(n.data[1])))},5,r)}function $a(e,t){return gs(e.subarray(Df(e,t&&t.dictionary),-4),{i:2},t&&t.out,t&&t.dictionary)}var If=function(){function e(t,r){this.o=yn.call(this,t,r)||{},this.G=Fa,this.I=it,this.Z=za}return e.prototype.i=function(){var t=this;this.s.ondata=function(r,n){t.ondata(r,n)}},e.prototype.push=function(t,r){if(this.ondata||K(5),this.s)this.s.push(t,r);else{if(this.p&&this.p.length){var n=new ee(this.p.length+t.length);n.set(this.p),n.set(t,this.p.length)}else this.p=t;this.p.length>2&&(this.s=this.p[0]==31&&this.p[1]==139&&this.p[2]==8?new this.G(this.o):(this.p[0]&15)!=8||this.p[0]>>4>7||(this.p[0]<<8|this.p[1])%31?new this.I(this.o):new this.Z(this.o),this.i(),this.s.push(this.p,r),this.p=null)}},e}();H.Decompress=If;var IT=function(){function e(t,r){If.call(this,t,r),this.queuedSize=0,this.G=xx,this.I=Pf,this.Z=wx}return e.prototype.i=function(){var t=this;this.s.ondata=function(r,n,i){t.ondata(r,n,i)},this.s.ondrain=function(r){t.queuedSize-=r,t.ondrain&&t.ondrain(r)}},e.prototype.push=function(t,r){this.queuedSize+=t.length,If.prototype.push.call(this,t,r)},e}();H.AsyncDecompress=IT;function _T(e,t,r){return r||(r=t,t={}),typeof r!="function"&&K(7),e[0]==31&&e[1]==139&&e[2]==8?Sx(e,t,r):(e[0]&15)!=8||e[0]>>4>7||(e[0]<<8|e[1])%31?Rf(e,t,r):Ex(e,t,r)}function CT(e,t){return e[0]==31&&e[1]==139&&e[2]==8?Ba(e,t):(e[0]&15)!=8||e[0]>>4>7||(e[0]<<8|e[1])%31?vi(e,t):$a(e,t)}var Of=function(e,t,r,n){for(var i in e){var s=e[i],o=t+i,a=n;Array.isArray(s)&&(a=ys(n,s[1]),s=s[0]),ArrayBuffer.isView(s)?r[o]=[s,a]:(r[o+="/"]=[new ee(0),a],Of(s,o,r,n))}},Wv=typeof TextEncoder<"u"&&new TextEncoder,_f=typeof TextDecoder<"u"&&new TextDecoder,bx=0;try{_f.decode(cr,{stream:!0}),bx=1}catch{}var Ix=function(e){for(var t="",r=0;;){var n=e[r++],i=(n>127)+(n>223)+(n>239);if(r+i>e.length)return{s:t,r:Ot(e,r-1)};i?i==3?(n=((n&15)<<18|(e[r++]&63)<<12|(e[r++]&63)<<6|e[r++]&63)-65536,t+=String.fromCharCode(55296|n>>10,56320|n&1023)):i&1?t+=String.fromCharCode((n&31)<<6|e[r++]&63):t+=String.fromCharCode((n&15)<<12|(e[r++]&63)<<6|e[r++]&63):t+=String.fromCharCode(n)}},AT=function(){function e(t){this.ondata=t,bx?this.t=new TextDecoder:this.p=cr}return e.prototype.push=function(t,r){if(this.ondata||K(5),r=!!r,this.t){this.ondata(this.t.decode(t,{stream:!0}),r),r&&(this.t.decode().length&&K(8),this.t=null);return}this.p||K(4);var n=new ee(this.p.length+t.length);n.set(this.p),n.set(t,this.p.length);var i=Ix(n),s=i.s,o=i.r;r?(o.length&&K(8),this.p=null):this.p=o,this.ondata(s,r)},e}();H.DecodeUTF8=AT;var MT=function(){function e(t){this.ondata=t}return e.prototype.push=function(t,r){this.ondata||K(5),this.d&&K(4),this.ondata(Yr(t),this.d=r||!1)},e}();H.EncodeUTF8=MT;function Yr(e,t){if(t){for(var r=new ee(e.length),n=0;n<e.length;++n)r[n]=e.charCodeAt(n);return r}if(Wv)return Wv.encode(e);for(var i=e.length,s=new ee(e.length+(e.length>>1)),o=0,a=function(l){s[o++]=l},n=0;n<i;++n){if(o+5>s.length){var c=new ee(o+8+(i-n<<1));c.set(s),s=c}var u=e.charCodeAt(n);u<128||t?a(u):u<2048?(a(192|u>>6),a(128|u&63)):u>55295&&u<57344?(u=65536+(u&1047552)|e.charCodeAt(++n)&1023,a(240|u>>18),a(128|u>>12&63),a(128|u>>6&63),a(128|u&63)):(a(224|u>>12),a(128|u>>6&63),a(128|u&63))}return Ot(s,0,o)}function Uf(e,t){if(t){for(var r="",n=0;n<e.length;n+=16384)r+=String.fromCharCode.apply(null,e.subarray(n,n+16384));return r}else{if(_f)return _f.decode(e);var i=Ix(e),s=i.s,r=i.r;return r.length&&K(8),s}}var _x=function(e){return e==1?3:e<6?2:e==9?1:0},Cx=function(e,t){return t+30+rt(e,t+26)+rt(e,t+28)},Ax=function(e,t,r){var n=rt(e,t+28),i=rt(e,t+30),s=Uf(e.subarray(t+46,t+46+n),!(rt(e,t+8)&2048)),o=t+46+n,a=Mx(e,o,i,r,Ie(e,t+20),Ie(e,t+24),Ie(e,t+42)),c=a[0],u=a[1],l=a[2];return[rt(e,t+10),c,u,s,o+i+rt(e,t+32),l]},Mx=function(e,t,r,n,i,s,o){var a=i==4294967295,c=s==4294967295,u=o==4294967295,l=t+r,f=a+c+u;if(n&&f){for(;t+4<l;t+=4+rt(e,t+2))if(rt(e,t)==1)return[a?vf(e,t+4+8*c):i,c?vf(e,t+4):s,u?vf(e,t+4+8*(c+a)):o,1];n<2&&K(13)}return[i,s,o,0]},Kr=function(e){var t=0;if(e)for(var r in e){var n=e[r].length;n>65535&&K(9),t+=n+4}return t},ui=function(e,t,r,n,i,s,o,a){var c=n.length,u=r.extra,l=a&&a.length,f=Kr(u);he(e,t,o!=null?33639248:67324752),t+=4,o!=null&&(e[t++]=20,e[t++]=r.os),e[t]=20,t+=2,e[t++]=r.flag<<1|(s<0&&8),e[t++]=i&&8,e[t++]=r.compression&255,e[t++]=r.compression>>8;var h=new Date(r.mtime==null?Date.now():r.mtime),p=h.getFullYear()-1980;if((p<0||p>119)&&K(10),he(e,t,p<<25|h.getMonth()+1<<21|h.getDate()<<16|h.getHours()<<11|h.getMinutes()<<5|h.getSeconds()>>1),t+=4,s!=-1&&(he(e,t,r.crc),he(e,t+4,s<0?-s-2:s),he(e,t+8,r.size)),he(e,t+12,c),he(e,t+14,f),t+=16,o!=null&&(he(e,t,l),he(e,t+6,r.attrs),he(e,t+10,o),t+=14),e.set(n,t),t+=c,f)for(var d in u){var m=u[d],g=m.length;he(e,t,+d),he(e,t+2,g),e.set(m,t+4),t+=4+g}return l&&(e.set(a,t),t+=l),t},Lf=function(e,t,r,n,i){he(e,t,101010256),he(e,t+8,r),he(e,t+10,r),he(e,t+12,n),he(e,t+16,i)},ds=function(){function e(t){this.filename=t,this.c=pi(),this.size=0,this.compression=0}return e.prototype.process=function(t,r){this.ondata(null,t,r)},e.prototype.push=function(t,r){this.ondata||K(5),this.c.p(t),this.size+=t.length,r&&(this.crc=this.c.d()),this.process(t,r||!1)},e}();H.ZipPassThrough=ds;var TT=function(){function e(t,r){var n=this;r||(r={}),ds.call(this,t),this.d=new Ut(r,function(i,s){n.ondata(null,i,s)}),this.compression=8,this.flag=_x(r.level)}return e.prototype.process=function(t,r){try{this.d.push(t,r)}catch(n){this.ondata(n,null,r)}},e.prototype.push=function(t,r){ds.prototype.push.call(this,t,r)},e}();H.ZipDeflate=TT;var kT=function(){function e(t,r){var n=this;r||(r={}),ds.call(this,t),this.d=new gx(r,function(i,s,o){n.ondata(i,s,o)}),this.compression=8,this.flag=_x(r.level),this.terminate=this.d.terminate}return e.prototype.process=function(t,r){this.d.push(t,r)},e.prototype.push=function(t,r){ds.prototype.push.call(this,t,r)},e}();H.AsyncZipDeflate=kT;var NT=function(){function e(t){this.ondata=t,this.u=[],this.d=1}return e.prototype.add=function(t){var r=this;if(this.ondata||K(5),this.d&2)this.ondata(K(4+(this.d&1)*8,0,1),null,!1);else{var n=Yr(t.filename),i=n.length,s=t.comment,o=s&&Yr(s),a=i!=t.filename.length||o&&s.length!=o.length,c=i+Kr(t.extra)+30;i>65535&&this.ondata(K(11,0,1),null,!1);var u=new ee(c);ui(u,0,t,n,a,-1);var l=[u],f=function(){for(var g=0,y=l;g<y.length;g++){var I=y[g];r.ondata(null,I,!1)}l=[]},h=this.d;this.d=0;var p=this.u.length,d=ys(t,{f:n,u:a,o,t:function(){t.terminate&&t.terminate()},r:function(){if(f(),h){var g=r.u[p+1];g?g.r():r.d=1}h=1}}),m=0;t.ondata=function(g,y,I){if(g)r.ondata(g,y,I),r.terminate();else if(m+=y.length,l.push(y),I){var _=new ee(16);he(_,0,134695760),he(_,4,t.crc),he(_,8,m),he(_,12,t.size),l.push(_),d.c=m,d.b=c+m+16,d.crc=t.crc,d.size=t.size,h&&d.r(),h=1}else h&&f()},this.u.push(d)}},e.prototype.end=function(){var t=this;if(this.d&2){this.ondata(K(4+(this.d&1)*8,0,1),null,!0);return}this.d?this.e():this.u.push({r:function(){t.d&1&&(t.u.splice(-1,1),t.e())},t:function(){}}),this.d=3},e.prototype.e=function(){for(var t=0,r=0,n=0,i=0,s=this.u;i<s.length;i++){var o=s[i];n+=46+o.f.length+Kr(o.extra)+(o.o?o.o.length:0)}for(var a=new ee(n+22),c=0,u=this.u;c<u.length;c++){var o=u[c];ui(a,t,o,o.f,o.u,-o.c-2,r,o.o),t+=46+o.f.length+Kr(o.extra)+(o.o?o.o.length:0),r+=o.b}Lf(a,t,this.u.length,n,r),this.ondata(null,a,!0),this.d=2},e.prototype.terminate=function(){for(var t=0,r=this.u;t<r.length;t++){var n=r[t];n.t()}this.d=2},e}();H.Zip=NT;function DT(e,t,r){r||(r=t,t={}),typeof r!="function"&&K(7);var n={};Of(e,"",n,t);var i=Object.keys(n),s=i.length,o=0,a=0,c=s,u=new Array(s),l=[],f=function(){for(var g=0;g<l.length;++g)l[g]()},h=function(g,y){Va(function(){r(g,y)})};Va(function(){h=r});var p=function(){var g=new ee(a+22),y=o,I=a-o;a=0;for(var _=0;_<c;++_){var E=u[_];try{var b=E.c.length;ui(g,a,E,E.f,E.u,b);var C=30+E.f.length+Kr(E.extra),v=a+C;g.set(E.c,v),ui(g,o,E,E.f,E.u,b,a,E.m),o+=16+C+(E.m?E.m.length:0),a=v+b}catch(w){return h(w,null)}}Lf(g,o,u.length,I,y),h(null,g)};s||p();for(var d=function(g){var y=i[g],I=n[y],_=I[0],E=I[1],b=pi(),C=_.length;b.p(_);var v=Yr(y),w=v.length,x=E.comment,T=x&&Yr(x),M=T&&T.length,P=Kr(E.extra),F=E.level==0?0:8,S=function(O,R){if(O)f(),h(O,null);else{var N=R.length;u[g]=ys(E,{size:C,crc:b.d(),c:R,f:v,m:T,u:w!=y.length||T&&x.length!=M,compression:F}),o+=30+w+P+N,a+=76+2*(w+P)+(M||0)+N,--s||p()}};if(w>65535&&S(K(11,0,1),null),!F)S(null,_);else if(C<16e4)try{S(null,vs(_,E))}catch(O){S(O,null)}else l.push(yx(_,E,S))},m=0;m<c;++m)d(m);return f}function PT(e,t){t||(t={});var r={},n=[];Of(e,"",r,t);var i=0,s=0;for(var o in r){var a=r[o],c=a[0],u=a[1],l=u.level==0?0:8,f=Yr(o),h=f.length,p=u.comment,d=p&&Yr(p),m=d&&d.length,g=Kr(u.extra);h>65535&&K(11);var y=l?vs(c,u):c,I=y.length,_=pi();_.p(c),n.push(ys(u,{size:c.length,crc:_.d(),c:y,f,m:d,u:h!=o.length||d&&p.length!=m,o:i,compression:l})),i+=30+h+g+I,s+=76+2*(h+g)+(m||0)+I}for(var E=new ee(s+22),b=i,C=s-i,v=0;v<n.length;++v){var f=n[v];ui(E,f.o,f,f.f,f.u,f.c.length);var w=30+f.f.length+Kr(f.extra);E.set(f.c,f.o+w),ui(E,i,f,f.f,f.u,f.c.length,f.o,f.m),i+=16+w+(f.m?f.m.length:0)}return Lf(E,i,n.length,C,b),E}var Tx=function(){function e(){}return e.prototype.push=function(t,r){this.ondata(null,t,r)},e.compression=0,e}();H.UnzipPassThrough=Tx;var RT=function(){function e(){var t=this;this.i=new it(function(r,n){t.ondata(null,r,n)})}return e.prototype.push=function(t,r){try{this.i.push(t,r)}catch(n){this.ondata(n,null,r)}},e.compression=8,e}();H.UnzipInflate=RT;var OT=function(){function e(t,r){var n=this;r<32e4?this.i=new it(function(i,s){n.ondata(null,i,s)}):(this.i=new Pf(function(i,s,o){n.ondata(i,s,o)}),this.terminate=this.i.terminate)}return e.prototype.push=function(t,r){this.i.terminate&&(t=Ot(t,0)),this.i.push(t,r)},e.compression=8,e}();H.AsyncUnzipInflate=OT;var UT=function(){function e(t){this.onfile=t,this.k=[],this.o={0:Tx},this.p=cr}return e.prototype.push=function(t,r){var n=this;if(this.onfile||K(5),this.p||K(4),this.c>0){var i=Math.min(this.c,t.length),s=t.subarray(0,i);if(this.c-=i,this.d?this.d.push(s,!this.c):this.k[0].push(s),t=t.subarray(i),t.length)return this.push(t,r)}else{var o=0,a=0,c=void 0,u=void 0;this.p.length?t.length?(u=new ee(this.p.length+t.length),u.set(this.p),u.set(t,this.p.length)):u=this.p:u=t;for(var l=u.length,f=this.c,h=f&&this.d,p=function(){var y=Ie(u,a);if(y==67324752){o=1,c=a,d.d=null,d.c=0;var I=rt(u,a+6),_=rt(u,a+8),E=I&2048,b=I&8,C=rt(u,a+26),v=rt(u,a+28);if(l>a+30+C+v){var w=[];d.k.unshift(w),o=2;var x=Ie(u,a+18),T=Ie(u,a+22),M=Uf(u.subarray(a+30,a+=30+C),!E),P=Mx(u,a,v,2,x,T,0),F=P[0],S=P[1],O=P[3];b&&(F=-1-O),a+=v,d.c=F;var R,N={name:M,compression:_,start:function(){if(N.ondata||K(5),!F)N.ondata(null,cr,!0);else{var U=n.o[_];U||N.ondata(K(14,"unknown compression type "+_,1),null,!1),R=F<0?new U(M):new U(M,F,S),R.ondata=function(Y,J,de){N.ondata(Y,J,de)};for(var L=0,j=w;L<j.length;L++){var V=j[L];R.push(V,!1)}n.k[0]==w&&n.c?n.d=R:R.push(cr,!0)}},terminate:function(){R&&R.terminate&&R.terminate()}};F>=0&&(N.size=F,N.originalSize=S),d.onfile(N)}return"break"}else if(f){if(y==134695760)return c=a+=12+(f==-2&&8),o=3,d.c=0,"break";if(y==33639248)return c=a-=4,o=3,d.c=0,"break"}},d=this;a<l-4;++a){var m=p();if(m==="break")break}if(this.p=cr,f<0){var g=o?u.subarray(0,c-12-(f==-2&&8)-(Ie(u,c-16)==134695760&&4)):u.subarray(0,a);h?h.push(g,!!o):this.k[+(o==2)].push(g)}if(o&2)return this.push(u.subarray(a),r);this.p=u.subarray(a)}r&&(this.c&&K(13),this.p=null)},e.prototype.register=function(t){this.o[t.compression]=t},e}();H.Unzip=UT;var Va=typeof queueMicrotask=="function"?queueMicrotask:typeof setTimeout=="function"?setTimeout:function(e){e()};function LT(e,t,r){r||(r=t,t={}),typeof r!="function"&&K(7);var n=[],i=function(){for(var g=0;g<n.length;++g)n[g]()},s={},o=function(g,y){Va(function(){r(g,y)})};Va(function(){o=r});for(var a=e.length-22;Ie(e,a)!=101010256;--a)if(!a||e.length-a>65558)return o(K(13,0,1),null),i;var c=rt(e,a+8);if(c){var u=c,l=Ie(e,a+16),f=Ie(e,a-20)==117853008;if(f){var h=Ie(e,a-12);f=Ie(e,h)==101075792,f&&(u=c=Ie(e,h+32),l=Ie(e,h+48))}for(var p=t&&t.filter,d=function(g){var y=Ax(e,l,f),I=y[0],_=y[1],E=y[2],b=y[3],C=y[4],v=y[5],w=Cx(e,v);l=C;var x=function(M,P){M?(i(),o(M,null)):(P&&(s[b]=P),--c||o(null,s))};if(!p||p({name:b,size:_,originalSize:E,compression:I}))if(!I)x(null,Ot(e,w,w+_));else if(I==8){var T=e.subarray(w,w+_);if(E<524288||_>.8*E)try{x(null,vi(T,{out:new ee(E)}))}catch(M){x(M,null)}else n.push(Rf(T,{size:E},x))}else x(K(14,"unknown compression type "+I,1),null);else x(null,null)},m=0;m<u;++m)d(m)}else o(null,{});return i}function FT(e,t){for(var r={},n=e.length-22;Ie(e,n)!=101010256;--n)(!n||e.length-n>65558)&&K(13);var i=rt(e,n+8);if(!i)return{};var s=Ie(e,n+16),o=Ie(e,n-20)==117853008;if(o){var a=Ie(e,n-12);o=Ie(e,a)==101075792,o&&(i=Ie(e,a+32),s=Ie(e,a+48))}for(var c=t&&t.filter,u=0;u<i;++u){var l=Ax(e,s,o),f=l[0],h=l[1],p=l[2],d=l[3],m=l[4],g=l[5],y=Cx(e,g);s=m,(!c||c({name:d,size:h,originalSize:p,compression:f}))&&(f?f==8?r[d]=vi(e.subarray(y,y+h),{out:new ee(p)}):K(14,"unknown compression type "+f):r[d]=Ot(e,y,y+h))}return r}});var Bf=q(Ka=>{"use strict";Object.defineProperty(Ka,"__esModule",{value:!0});Ka.NIFTIEXTENSION=void 0;var Ff=class{esize;ecode;edata;littleEndian;constructor(t,r,n,i){if(t%16!=0)throw new Error("This does not appear to be a NIFTI extension");this.esize=t,this.ecode=r,this.edata=n,this.littleEndian=i}toArrayBuffer(){let t=new Uint8Array(this.esize),r=new Uint8Array(this.edata);t.set(r,8);let n=new DataView(t.buffer);return n.setInt32(0,this.esize,this.littleEndian),n.setInt32(4,this.ecode,this.littleEndian),t.buffer}};Ka.NIFTIEXTENSION=Ff});var xs=q(Ya=>{"use strict";Object.defineProperty(Ya,"__esModule",{value:!0});Ya.Utils=void 0;var BT=Bf(),zf=class e{static crcTable=null;static GUNZIP_MAGIC_COOKIE1=31;static GUNZIP_MAGIC_COOKIE2=139;static getStringAt(t,r,n){var i="",s,o;for(s=r;s<n;s+=1)o=t.getUint8(s),o!==0&&(i+=String.fromCharCode(o));return i}static getByteAt=function(t,r){return t.getUint8(r)};static getShortAt=function(t,r,n){return t.getInt16(r,n)};static getIntAt(t,r,n){return t.getInt32(r,n)}static getFloatAt(t,r,n){return t.getFloat32(r,n)}static getDoubleAt(t,r,n){return t.getFloat64(r,n)}static getInt64At(t,r,n){let i=t.getUint32(r,n),s=t.getInt32(r+4,n),o;return n?o=s*2**32+i:o=i*2**32+s,s<0&&(o+=-1*2**32*2**32),o}static getExtensionsAt(t,r,n,i){let s=[],o=r;for(;o<i;){let a=n,c=e.getIntAt(t,o,n);if(!c)break;if(c+o>i&&(a=!a,c=e.getIntAt(t,o,a),c+o>i))throw new Error("This does not appear to be a valid NIFTI extension");if(c%16!=0)throw new Error("This does not appear to be a NIFTI extension");let u=e.getIntAt(t,o+4,a),l=t.buffer.slice(o+8,o+c);console.log("extensionByteIndex: "+(o+8)+" esize: "+c),console.log(l);let f=new BT.NIFTIEXTENSION(c,u,l,a);s.push(f),o+=c}return s}static toArrayBuffer(t){var r,n,i;for(r=new ArrayBuffer(t.length),n=new Uint8Array(r),i=0;i<t.length;i+=1)n[i]=t[i];return r}static isString(t){return typeof t=="string"||t instanceof String}static formatNumber(t,r=void 0){let n;return e.isString(t)?n=Number(t):n=t,r?n=n.toPrecision(5):n=n.toPrecision(7),parseFloat(n)}static makeCRCTable(){let t,r=[];for(var n=0;n<256;n++){t=n;for(var i=0;i<8;i++)t=t&1?3988292384^t>>>1:t>>>1;r[n]=t}return r}static crc32(t){e.crcTable||(e.crcTable=e.makeCRCTable());let r=e.crcTable,n=-1;for(var i=0;i<t.byteLength;i++)n=n>>>8^r[(n^t.getUint8(i))&255];return(n^-1)>>>0}};Ya.Utils=zf});var Ha=q(Ja=>{"use strict";Object.defineProperty(Ja,"__esModule",{value:!0});Ja.NIFTI1=void 0;var Q=xs(),$f=class e{littleEndian=!1;dim_info=0;dims=[];intent_p1=0;intent_p2=0;intent_p3=0;intent_code=0;datatypeCode=0;numBitsPerVoxel=0;slice_start=0;slice_end=0;slice_code=0;pixDims=[];vox_offset=0;scl_slope=1;scl_inter=0;xyzt_units=0;cal_max=0;cal_min=0;slice_duration=0;toffset=0;description="";aux_file="";intent_name="";qform_code=0;sform_code=0;quatern_a=0;quatern_b=0;quatern_c=0;quatern_d=0;qoffset_x=0;qoffset_y=0;qoffset_z=0;affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];qfac=1;quatern_R;magic="0";isHDR=!1;extensionFlag=[0,0,0,0];extensionSize=0;extensionCode=0;extensions=[];static TYPE_NONE=0;static TYPE_BINARY=1;static TYPE_UINT8=2;static TYPE_INT16=4;static TYPE_INT32=8;static TYPE_FLOAT32=16;static TYPE_COMPLEX64=32;static TYPE_FLOAT64=64;static TYPE_RGB24=128;static TYPE_INT8=256;static TYPE_UINT16=512;static TYPE_UINT32=768;static TYPE_INT64=1024;static TYPE_UINT64=1280;static TYPE_FLOAT128=1536;static TYPE_COMPLEX128=1792;static TYPE_COMPLEX256=2048;static XFORM_UNKNOWN=0;static XFORM_SCANNER_ANAT=1;static XFORM_ALIGNED_ANAT=2;static XFORM_TALAIRACH=3;static XFORM_MNI_152=4;static SPATIAL_UNITS_MASK=7;static TEMPORAL_UNITS_MASK=56;static UNITS_UNKNOWN=0;static UNITS_METER=1;static UNITS_MM=2;static UNITS_MICRON=3;static UNITS_SEC=8;static UNITS_MSEC=16;static UNITS_USEC=24;static UNITS_HZ=32;static UNITS_PPM=40;static UNITS_RADS=48;static MAGIC_COOKIE=348;static STANDARD_HEADER_SIZE=348;static MAGIC_NUMBER_LOCATION=344;static MAGIC_NUMBER=[110,43,49];static MAGIC_NUMBER2=[110,105,49];static EXTENSION_HEADER_SIZE=8;readHeader(t){var r=new DataView(t),n=Q.Utils.getIntAt(r,0,this.littleEndian),i,s,o,a;if(n!==e.MAGIC_COOKIE&&(this.littleEndian=!0,n=Q.Utils.getIntAt(r,0,this.littleEndian)),n!==e.MAGIC_COOKIE)throw new Error("This does not appear to be a NIFTI file!");for(this.dim_info=Q.Utils.getByteAt(r,39),i=0;i<8;i+=1)a=40+i*2,this.dims[i]=Q.Utils.getShortAt(r,a,this.littleEndian);for(this.intent_p1=Q.Utils.getFloatAt(r,56,this.littleEndian),this.intent_p2=Q.Utils.getFloatAt(r,60,this.littleEndian),this.intent_p3=Q.Utils.getFloatAt(r,64,this.littleEndian),this.intent_code=Q.Utils.getShortAt(r,68,this.littleEndian),this.datatypeCode=Q.Utils.getShortAt(r,70,this.littleEndian),this.numBitsPerVoxel=Q.Utils.getShortAt(r,72,this.littleEndian),this.slice_start=Q.Utils.getShortAt(r,74,this.littleEndian),i=0;i<8;i+=1)a=76+i*4,this.pixDims[i]=Q.Utils.getFloatAt(r,a,this.littleEndian);if(this.vox_offset=Q.Utils.getFloatAt(r,108,this.littleEndian),this.scl_slope=Q.Utils.getFloatAt(r,112,this.littleEndian),this.scl_inter=Q.Utils.getFloatAt(r,116,this.littleEndian),this.slice_end=Q.Utils.getShortAt(r,120,this.littleEndian),this.slice_code=Q.Utils.getByteAt(r,122),this.xyzt_units=Q.Utils.getByteAt(r,123),this.cal_max=Q.Utils.getFloatAt(r,124,this.littleEndian),this.cal_min=Q.Utils.getFloatAt(r,128,this.littleEndian),this.slice_duration=Q.Utils.getFloatAt(r,132,this.littleEndian),this.toffset=Q.Utils.getFloatAt(r,136,this.littleEndian),this.description=Q.Utils.getStringAt(r,148,228),this.aux_file=Q.Utils.getStringAt(r,228,252),this.qform_code=Q.Utils.getShortAt(r,252,this.littleEndian),this.sform_code=Q.Utils.getShortAt(r,254,this.littleEndian),this.quatern_b=Q.Utils.getFloatAt(r,256,this.littleEndian),this.quatern_c=Q.Utils.getFloatAt(r,260,this.littleEndian),this.quatern_d=Q.Utils.getFloatAt(r,264,this.littleEndian),this.quatern_a=Math.sqrt(1-(Math.pow(this.quatern_b,2)+Math.pow(this.quatern_c,2)+Math.pow(this.quatern_d,2))),this.qoffset_x=Q.Utils.getFloatAt(r,268,this.littleEndian),this.qoffset_y=Q.Utils.getFloatAt(r,272,this.littleEndian),this.qoffset_z=Q.Utils.getFloatAt(r,276,this.littleEndian),this.qform_code<1&&this.sform_code<1&&(this.affine[0][0]=this.pixDims[1],this.affine[1][1]=this.pixDims[2],this.affine[2][2]=this.pixDims[3]),this.qform_code>0&&this.sform_code<this.qform_code){let c=this.quatern_a,u=this.quatern_b,l=this.quatern_c,f=this.quatern_d;for(this.qfac=this.pixDims[0]===0?1:this.pixDims[0],this.quatern_R=[[c*c+u*u-l*l-f*f,2*u*l-2*c*f,2*u*f+2*c*l],[2*u*l+2*c*f,c*c+l*l-u*u-f*f,2*l*f-2*c*u],[2*u*f-2*c*l,2*l*f+2*c*u,c*c+f*f-l*l-u*u]],s=0;s<3;s+=1)for(o=0;o<3;o+=1)this.affine[s][o]=this.quatern_R[s][o]*this.pixDims[o+1],o===2&&(this.affine[s][o]*=this.qfac);this.affine[0][3]=this.qoffset_x,this.affine[1][3]=this.qoffset_y,this.affine[2][3]=this.qoffset_z}else if(this.sform_code>0)for(s=0;s<3;s+=1)for(o=0;o<4;o+=1)a=280+(s*4+o)*4,this.affine[s][o]=Q.Utils.getFloatAt(r,a,this.littleEndian);if(this.affine[3][0]=0,this.affine[3][1]=0,this.affine[3][2]=0,this.affine[3][3]=1,this.intent_name=Q.Utils.getStringAt(r,328,344),this.magic=Q.Utils.getStringAt(r,344,348),this.isHDR=this.magic===String.fromCharCode.apply(null,e.MAGIC_NUMBER2),r.byteLength>e.MAGIC_COOKIE){this.extensionFlag[0]=Q.Utils.getByteAt(r,348),this.extensionFlag[1]=Q.Utils.getByteAt(r,349),this.extensionFlag[2]=Q.Utils.getByteAt(r,350),this.extensionFlag[3]=Q.Utils.getByteAt(r,351);let c=!0;!this.isHDR&&this.vox_offset<=352&&(c=!1),r.byteLength<=368&&(c=!1),c&&this.extensionFlag[0]&&(this.extensions=Q.Utils.getExtensionsAt(r,this.getExtensionLocation(),this.littleEndian,this.vox_offset),this.extensionSize=this.extensions[0].esize,this.extensionCode=this.extensions[0].ecode)}}toFormattedString(){var t=Q.Utils.formatNumber,r="";return r+="Dim Info = "+this.dim_info+`
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
`),r}getDatatypeCodeString=function(t){return t===e.TYPE_UINT8?"1-Byte Unsigned Integer":t===e.TYPE_INT16?"2-Byte Signed Integer":t===e.TYPE_INT32?"4-Byte Signed Integer":t===e.TYPE_FLOAT32?"4-Byte Float":t===e.TYPE_FLOAT64?"8-Byte Float":t===e.TYPE_RGB24?"RGB":t===e.TYPE_INT8?"1-Byte Signed Integer":t===e.TYPE_UINT16?"2-Byte Unsigned Integer":t===e.TYPE_UINT32?"4-Byte Unsigned Integer":t===e.TYPE_INT64?"8-Byte Signed Integer":t===e.TYPE_UINT64?"8-Byte Unsigned Integer":"Unknown"};getTransformCodeString=function(t){return t===e.XFORM_SCANNER_ANAT?"Scanner":t===e.XFORM_ALIGNED_ANAT?"Aligned":t===e.XFORM_TALAIRACH?"Talairach":t===e.XFORM_MNI_152?"MNI":"Unknown"};getUnitsCodeString=function(t){return t===e.UNITS_METER?"Meters":t===e.UNITS_MM?"Millimeters":t===e.UNITS_MICRON?"Microns":t===e.UNITS_SEC?"Seconds":t===e.UNITS_MSEC?"Milliseconds":t===e.UNITS_USEC?"Microseconds":t===e.UNITS_HZ?"Hz":t===e.UNITS_PPM?"PPM":t===e.UNITS_RADS?"Rads":"Unknown"};getQformMat(){return this.convertNiftiQFormToNiftiSForm(this.quatern_b,this.quatern_c,this.quatern_d,this.qoffset_x,this.qoffset_y,this.qoffset_z,this.pixDims[1],this.pixDims[2],this.pixDims[3],this.pixDims[0])}convertNiftiQFormToNiftiSForm(t,r,n,i,s,o,a,c,u,l){var f=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],h,p=t,d=r,m=n,g,y,I;return f[3][0]=f[3][1]=f[3][2]=0,f[3][3]=1,h=1-(p*p+d*d+m*m),h<1e-7?(h=1/Math.sqrt(p*p+d*d+m*m),p*=h,d*=h,m*=h,h=0):h=Math.sqrt(h),g=a>0?a:1,y=c>0?c:1,I=u>0?u:1,l<0&&(I=-I),f[0][0]=(h*h+p*p-d*d-m*m)*g,f[0][1]=2*(p*d-h*m)*y,f[0][2]=2*(p*m+h*d)*I,f[1][0]=2*(p*d+h*m)*g,f[1][1]=(h*h+d*d-p*p-m*m)*y,f[1][2]=2*(d*m-h*p)*I,f[2][0]=2*(p*m-h*d)*g,f[2][1]=2*(d*m+h*p)*y,f[2][2]=(h*h+m*m-d*d-p*p)*I,f[0][3]=i,f[1][3]=s,f[2][3]=o,f}convertNiftiSFormToNEMA(t){var r,n,i,s,o,a,c,u,l,f,h,p,d,m,g,y,I,_,E,b,C,v,w,x,T,M,P,F,S,O,R,N,U,L;if(g=0,P=[[0,0,0],[0,0,0],[0,0,0]],F=[[0,0,0],[0,0,0],[0,0,0]],r=t[0][0],n=t[0][1],i=t[0][2],s=t[1][0],o=t[1][1],a=t[1][2],c=t[2][0],u=t[2][1],l=t[2][2],f=Math.sqrt(r*r+s*s+c*c),f===0||(r/=f,s/=f,c/=f,f=Math.sqrt(n*n+o*o+u*u),f===0))return null;if(n/=f,o/=f,u/=f,f=r*n+s*o+c*u,Math.abs(f)>1e-4){if(n-=f*r,o-=f*s,u-=f*c,f=Math.sqrt(n*n+o*o+u*u),f===0)return null;n/=f,o/=f,u/=f}if(f=Math.sqrt(i*i+a*a+l*l),f===0?(i=s*u-c*o,a=c*n-u*r,l=r*o-s*n):(i/=f,a/=f,l/=f),f=r*i+s*a+c*l,Math.abs(f)>1e-4){if(i-=f*r,a-=f*s,l-=f*c,f=Math.sqrt(i*i+a*a+l*l),f===0)return null;i/=f,a/=f,l/=f}if(f=n*i+o*a+u*l,Math.abs(f)>1e-4){if(i-=f*n,a-=f*o,l-=f*u,f=Math.sqrt(i*i+a*a+l*l),f===0)return null;i/=f,a/=f,l/=f}if(P[0][0]=r,P[0][1]=n,P[0][2]=i,P[1][0]=s,P[1][1]=o,P[1][2]=a,P[2][0]=c,P[2][1]=u,P[2][2]=l,h=this.nifti_mat33_determ(P),h===0)return null;for(M=-666,E=v=w=x=1,b=2,C=3,d=1;d<=3;d+=1)for(m=1;m<=3;m+=1)if(d!==m){for(g=1;g<=3;g+=1)if(!(d===g||m===g))for(F[0][0]=F[0][1]=F[0][2]=F[1][0]=F[1][1]=F[1][2]=F[2][0]=F[2][1]=F[2][2]=0,y=-1;y<=1;y+=2)for(I=-1;I<=1;I+=2)for(_=-1;_<=1;_+=2)F[0][d-1]=y,F[1][m-1]=I,F[2][g-1]=_,p=this.nifti_mat33_determ(F),p*h>0&&(T=this.nifti_mat33_mul(F,P),f=T[0][0]+T[1][1]+T[2][2],f>M&&(M=f,E=d,b=m,C=g,v=y,w=I,x=_))}switch(S=O=R=N=U=L="",E*v){case 1:S="X",N="+";break;case-1:S="X",N="-";break;case 2:S="Y",N="+";break;case-2:S="Y",N="-";break;case 3:S="Z",N="+";break;case-3:S="Z",N="-";break}switch(b*w){case 1:O="X",U="+";break;case-1:O="X",U="-";break;case 2:O="Y",U="+";break;case-2:O="Y",U="-";break;case 3:O="Z",U="+";break;case-3:O="Z",U="-";break}switch(C*x){case 1:R="X",L="+";break;case-1:R="X",L="-";break;case 2:R="Y",L="+";break;case-2:R="Y",L="-";break;case 3:R="Z",L="+";break;case-3:R="Z",L="-";break}return S+O+R+N+U+L}nifti_mat33_mul=function(t,r){var n=[[0,0,0],[0,0,0],[0,0,0]],i,s;for(i=0;i<3;i+=1)for(s=0;s<3;s+=1)n[i][s]=t[i][0]*r[0][s]+t[i][1]*r[1][s]+t[i][2]*r[2][s];return n};nifti_mat33_determ=function(t){var r,n,i,s,o,a,c,u,l;return r=t[0][0],n=t[0][1],i=t[0][2],s=t[1][0],o=t[1][1],a=t[1][2],c=t[2][0],u=t[2][1],l=t[2][2],r*o*l-r*u*a-s*n*l+s*u*i+c*n*a-c*o*i};getExtensionLocation(){return e.MAGIC_COOKIE+4}getExtensionSize(t){return Q.Utils.getIntAt(t,this.getExtensionLocation(),this.littleEndian)}getExtensionCode(t){return Q.Utils.getIntAt(t,this.getExtensionLocation()+4,this.littleEndian)}addExtension(t,r=-1){r==-1?this.extensions.push(t):this.extensions.splice(r,0,t),this.vox_offset+=t.esize}removeExtension(t){let r=this.extensions[t];r&&(this.vox_offset-=r.esize),this.extensions.splice(t,1)}toArrayBuffer(t=!1){let i=352;if(t)for(let c of this.extensions)i+=c.esize;let s=new Uint8Array(i),o=new DataView(s.buffer);o.setInt32(0,348,this.littleEndian),o.setUint8(39,this.dim_info);for(let c=0;c<8;c++)o.setUint16(40+2*c,this.dims[c],this.littleEndian);o.setFloat32(56,this.intent_p1,this.littleEndian),o.setFloat32(60,this.intent_p2,this.littleEndian),o.setFloat32(64,this.intent_p3,this.littleEndian),o.setInt16(68,this.intent_code,this.littleEndian),o.setInt16(70,this.datatypeCode,this.littleEndian),o.setInt16(72,this.numBitsPerVoxel,this.littleEndian),o.setInt16(74,this.slice_start,this.littleEndian);for(let c=0;c<8;c++)o.setFloat32(76+4*c,this.pixDims[c],this.littleEndian);o.setFloat32(108,this.vox_offset,this.littleEndian),o.setFloat32(112,this.scl_slope,this.littleEndian),o.setFloat32(116,this.scl_inter,this.littleEndian),o.setInt16(120,this.slice_end,this.littleEndian),o.setUint8(122,this.slice_code),o.setUint8(123,this.xyzt_units),o.setFloat32(124,this.cal_max,this.littleEndian),o.setFloat32(128,this.cal_min,this.littleEndian),o.setFloat32(132,this.slice_duration,this.littleEndian),o.setFloat32(136,this.toffset,this.littleEndian),s.set(Buffer.from(this.description),148),s.set(Buffer.from(this.aux_file),228),o.setInt16(252,this.qform_code,this.littleEndian),o.setInt16(254,this.sform_code,this.littleEndian),o.setFloat32(256,this.quatern_b,this.littleEndian),o.setFloat32(260,this.quatern_c,this.littleEndian),o.setFloat32(264,this.quatern_d,this.littleEndian),o.setFloat32(268,this.qoffset_x,this.littleEndian),o.setFloat32(272,this.qoffset_y,this.littleEndian),o.setFloat32(276,this.qoffset_z,this.littleEndian);let a=this.affine.flat();for(let c=0;c<12;c++)o.setFloat32(280+4*c,a[c],this.littleEndian);if(s.set(Buffer.from(this.intent_name),328),s.set(Buffer.from(this.magic),344),t){s.set(Uint8Array.from([1,0,0,0]),348);let c=this.getExtensionLocation();for(let u of this.extensions)o.setInt32(c,u.esize,u.littleEndian),o.setInt32(c+4,u.ecode,u.littleEndian),s.set(new Uint8Array(u.edata),c+8),c+=u.esize}else s.set(new Uint8Array(4).fill(0),348);return s.buffer}};Ja.NIFTI1=$f});var Gf=q(Wa=>{"use strict";Object.defineProperty(Wa,"__esModule",{value:!0});Wa.NIFTI2=void 0;var ht=Ha(),te=xs(),Vf=class e{littleEndian=!1;dim_info=0;dims=[];intent_p1=0;intent_p2=0;intent_p3=0;intent_code=0;datatypeCode=0;numBitsPerVoxel=0;slice_start=0;slice_end=0;slice_code=0;pixDims=[];vox_offset=0;scl_slope=1;scl_inter=0;xyzt_units=0;cal_max=0;cal_min=0;slice_duration=0;toffset=0;description="";aux_file="";intent_name="";qform_code=0;sform_code=0;quatern_b=0;quatern_c=0;quatern_d=0;qoffset_x=0;qoffset_y=0;qoffset_z=0;affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];magic="0";extensionFlag=[0,0,0,0];extensions=[];extensionSize=0;extensionCode=0;static MAGIC_COOKIE=540;static MAGIC_NUMBER_LOCATION=4;static MAGIC_NUMBER=[110,43,50,0,13,10,26,10];static MAGIC_NUMBER2=[110,105,50,0,13,10,26,10];readHeader(t){var r=new DataView(t),n=te.Utils.getIntAt(r,0,this.littleEndian),i,s,o,a,c;if(n!==e.MAGIC_COOKIE&&(this.littleEndian=!0,n=te.Utils.getIntAt(r,0,this.littleEndian)),n!==e.MAGIC_COOKIE)throw new Error("This does not appear to be a NIFTI file!");for(this.magic=te.Utils.getStringAt(r,4,12),this.datatypeCode=te.Utils.getShortAt(r,12,this.littleEndian),this.numBitsPerVoxel=te.Utils.getShortAt(r,14,this.littleEndian),i=0;i<8;i+=1)a=16+i*8,this.dims[i]=te.Utils.getInt64At(r,a,this.littleEndian);for(this.intent_p1=te.Utils.getDoubleAt(r,80,this.littleEndian),this.intent_p2=te.Utils.getDoubleAt(r,88,this.littleEndian),this.intent_p3=te.Utils.getDoubleAt(r,96,this.littleEndian),i=0;i<8;i+=1)a=104+i*8,this.pixDims[i]=te.Utils.getDoubleAt(r,a,this.littleEndian);for(this.vox_offset=te.Utils.getInt64At(r,168,this.littleEndian),this.scl_slope=te.Utils.getDoubleAt(r,176,this.littleEndian),this.scl_inter=te.Utils.getDoubleAt(r,184,this.littleEndian),this.cal_max=te.Utils.getDoubleAt(r,192,this.littleEndian),this.cal_min=te.Utils.getDoubleAt(r,200,this.littleEndian),this.slice_duration=te.Utils.getDoubleAt(r,208,this.littleEndian),this.toffset=te.Utils.getDoubleAt(r,216,this.littleEndian),this.slice_start=te.Utils.getInt64At(r,224,this.littleEndian),this.slice_end=te.Utils.getInt64At(r,232,this.littleEndian),this.description=te.Utils.getStringAt(r,240,320),this.aux_file=te.Utils.getStringAt(r,320,344),this.qform_code=te.Utils.getIntAt(r,344,this.littleEndian),this.sform_code=te.Utils.getIntAt(r,348,this.littleEndian),this.quatern_b=te.Utils.getDoubleAt(r,352,this.littleEndian),this.quatern_c=te.Utils.getDoubleAt(r,360,this.littleEndian),this.quatern_d=te.Utils.getDoubleAt(r,368,this.littleEndian),this.qoffset_x=te.Utils.getDoubleAt(r,376,this.littleEndian),this.qoffset_y=te.Utils.getDoubleAt(r,384,this.littleEndian),this.qoffset_z=te.Utils.getDoubleAt(r,392,this.littleEndian),s=0;s<3;s+=1)for(o=0;o<4;o+=1)a=400+(s*4+o)*8,this.affine[s][o]=te.Utils.getDoubleAt(r,a,this.littleEndian);this.affine[3][0]=0,this.affine[3][1]=0,this.affine[3][2]=0,this.affine[3][3]=1,this.slice_code=te.Utils.getIntAt(r,496,this.littleEndian),this.xyzt_units=te.Utils.getIntAt(r,500,this.littleEndian),this.intent_code=te.Utils.getIntAt(r,504,this.littleEndian),this.intent_name=te.Utils.getStringAt(r,508,524),this.dim_info=te.Utils.getByteAt(r,524),r.byteLength>e.MAGIC_COOKIE&&(this.extensionFlag[0]=te.Utils.getByteAt(r,540),this.extensionFlag[1]=te.Utils.getByteAt(r,541),this.extensionFlag[2]=te.Utils.getByteAt(r,542),this.extensionFlag[3]=te.Utils.getByteAt(r,543),this.extensionFlag[0]&&(this.extensions=te.Utils.getExtensionsAt(r,this.getExtensionLocation(),this.littleEndian,this.vox_offset),this.extensionSize=this.extensions[0].esize,this.extensionCode=this.extensions[0].ecode))}toFormattedString(){var t=te.Utils.formatNumber,r="";return r+="Datatype = "+ +this.datatypeCode+" ("+this.getDatatypeCodeString(this.datatypeCode)+`)
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
`,r}getExtensionLocation=function(){return e.MAGIC_COOKIE+4};getExtensionSize=ht.NIFTI1.prototype.getExtensionSize;getExtensionCode=ht.NIFTI1.prototype.getExtensionCode;addExtension=ht.NIFTI1.prototype.addExtension;removeExtension=ht.NIFTI1.prototype.removeExtension;getDatatypeCodeString=ht.NIFTI1.prototype.getDatatypeCodeString;getTransformCodeString=ht.NIFTI1.prototype.getTransformCodeString;getUnitsCodeString=ht.NIFTI1.prototype.getUnitsCodeString;getQformMat=ht.NIFTI1.prototype.getQformMat;convertNiftiQFormToNiftiSForm=ht.NIFTI1.prototype.convertNiftiQFormToNiftiSForm;convertNiftiSFormToNEMA=ht.NIFTI1.prototype.convertNiftiSFormToNEMA;nifti_mat33_mul=ht.NIFTI1.prototype.nifti_mat33_mul;nifti_mat33_determ=ht.NIFTI1.prototype.nifti_mat33_determ;toArrayBuffer(t=!1){let i=544;if(t)for(let c of this.extensions)i+=c.esize;let s=new Uint8Array(i),o=new DataView(s.buffer);o.setInt32(0,540,this.littleEndian),s.set(Buffer.from(this.magic),4),o.setInt16(12,this.datatypeCode,this.littleEndian),o.setInt16(14,this.numBitsPerVoxel,this.littleEndian);for(let c=0;c<8;c++)o.setBigInt64(16+8*c,BigInt(this.dims[c]),this.littleEndian);o.setFloat64(80,this.intent_p1,this.littleEndian),o.setFloat64(88,this.intent_p2,this.littleEndian),o.setFloat64(96,this.intent_p3,this.littleEndian);for(let c=0;c<8;c++)o.setFloat64(104+8*c,this.pixDims[c],this.littleEndian);o.setBigInt64(168,BigInt(this.vox_offset),this.littleEndian),o.setFloat64(176,this.scl_slope,this.littleEndian),o.setFloat64(184,this.scl_inter,this.littleEndian),o.setFloat64(192,this.cal_max,this.littleEndian),o.setFloat64(200,this.cal_min,this.littleEndian),o.setFloat64(208,this.slice_duration,this.littleEndian),o.setFloat64(216,this.toffset,this.littleEndian),o.setBigInt64(224,BigInt(this.slice_start),this.littleEndian),o.setBigInt64(232,BigInt(this.slice_end),this.littleEndian),s.set(Buffer.from(this.description),240),s.set(Buffer.from(this.aux_file),320),o.setInt32(344,this.qform_code,this.littleEndian),o.setInt32(348,this.sform_code,this.littleEndian),o.setFloat64(352,this.quatern_b,this.littleEndian),o.setFloat64(360,this.quatern_c,this.littleEndian),o.setFloat64(368,this.quatern_d,this.littleEndian),o.setFloat64(376,this.qoffset_x,this.littleEndian),o.setFloat64(384,this.qoffset_y,this.littleEndian),o.setFloat64(392,this.qoffset_z,this.littleEndian);let a=this.affine.flat();for(let c=0;c<12;c++)o.setFloat64(400+8*c,a[c],this.littleEndian);if(o.setInt32(496,this.slice_code,this.littleEndian),o.setInt32(500,this.xyzt_units,this.littleEndian),o.setInt32(504,this.intent_code,this.littleEndian),s.set(Buffer.from(this.intent_name),508),o.setUint8(524,this.dim_info),t){s.set(Uint8Array.from([1,0,0,0]),540);let c=this.getExtensionLocation();for(let u of this.extensions)o.setInt32(c,u.esize,u.littleEndian),o.setInt32(c+4,u.ecode,u.littleEndian),s.set(new Uint8Array(u.edata),c+8),c+=u.esize}else s.set(new Uint8Array(4).fill(0),540);return s.buffer}};Wa.NIFTI2=Vf});var Rx=q(ae=>{"use strict";var zT=ae&&ae.__createBinding||(Object.create?function(e,t,r,n){n===void 0&&(n=r);var i=Object.getOwnPropertyDescriptor(t,r);(!i||("get"in i?!t.__esModule:i.writable||i.configurable))&&(i={enumerable:!0,get:function(){return t[r]}}),Object.defineProperty(e,n,i)}:function(e,t,r,n){n===void 0&&(n=r),e[n]=t[r]}),$T=ae&&ae.__setModuleDefault||(Object.create?function(e,t){Object.defineProperty(e,"default",{enumerable:!0,value:t})}:function(e,t){e.default=t}),VT=ae&&ae.__importStar||function(e){if(e&&e.__esModule)return e;var t={};if(e!=null)for(var r in e)r!=="default"&&Object.prototype.hasOwnProperty.call(e,r)&&zT(t,e,r);return $T(t,e),t};Object.defineProperty(ae,"__esModule",{value:!0});ae.readExtensionData=ae.readExtension=ae.readImage=ae.hasExtension=ae.readHeader=ae.decompress=ae.isCompressed=ae.isNIFTI=ae.isNIFTI2=ae.isNIFTI1=ae.NIFTIEXTENSION=ae.Utils=ae.NIFTI2=ae.NIFTI1=void 0;var GT=VT(kx()),Ft=Ha(),ur=Gf(),Nx=xs(),jT=Ha();Object.defineProperty(ae,"NIFTI1",{enumerable:!0,get:function(){return jT.NIFTI1}});var qT=Gf();Object.defineProperty(ae,"NIFTI2",{enumerable:!0,get:function(){return qT.NIFTI2}});var KT=xs();Object.defineProperty(ae,"Utils",{enumerable:!0,get:function(){return KT.Utils}});var YT=Bf();Object.defineProperty(ae,"NIFTIEXTENSION",{enumerable:!0,get:function(){return YT.NIFTIEXTENSION}});function jf(e,t=!1){var r,n,i,s;return e.byteLength<Ft.NIFTI1.STANDARD_HEADER_SIZE?!1:(r=new DataView(e),r&&(n=r.getUint8(Ft.NIFTI1.MAGIC_NUMBER_LOCATION)),i=r.getUint8(Ft.NIFTI1.MAGIC_NUMBER_LOCATION+1),s=r.getUint8(Ft.NIFTI1.MAGIC_NUMBER_LOCATION+2),t&&n===Ft.NIFTI1.MAGIC_NUMBER2[0]&&i===Ft.NIFTI1.MAGIC_NUMBER2[1]&&s===Ft.NIFTI1.MAGIC_NUMBER2[2]?!0:n===Ft.NIFTI1.MAGIC_NUMBER[0]&&i===Ft.NIFTI1.MAGIC_NUMBER[1]&&s===Ft.NIFTI1.MAGIC_NUMBER[2])}ae.isNIFTI1=jf;function qf(e,t=!1){var r,n,i,s;return e.byteLength<Ft.NIFTI1.STANDARD_HEADER_SIZE?!1:(r=new DataView(e),n=r.getUint8(ur.NIFTI2.MAGIC_NUMBER_LOCATION),i=r.getUint8(ur.NIFTI2.MAGIC_NUMBER_LOCATION+1),s=r.getUint8(ur.NIFTI2.MAGIC_NUMBER_LOCATION+2),t&&n===ur.NIFTI2.MAGIC_NUMBER2[0]&&i===ur.NIFTI2.MAGIC_NUMBER2[1]&&s===ur.NIFTI2.MAGIC_NUMBER2[2]?!0:n===ur.NIFTI2.MAGIC_NUMBER[0]&&i===ur.NIFTI2.MAGIC_NUMBER[1]&&s===ur.NIFTI2.MAGIC_NUMBER[2])}ae.isNIFTI2=qf;function JT(e,t=!1){return jf(e,t)||qf(e,t)}ae.isNIFTI=JT;function Dx(e){var t,r,n;return!!(e&&(t=new DataView(e),r=t.getUint8(0),n=t.getUint8(1),r===Nx.Utils.GUNZIP_MAGIC_COOKIE1||n===Nx.Utils.GUNZIP_MAGIC_COOKIE2))}ae.isCompressed=Dx;function Px(e){return GT.decompressSync(new Uint8Array(e)).buffer}ae.decompress=Px;function HT(e,t=!1){var r=null;return Dx(e)&&(e=Px(e)),jf(e,t)?r=new Ft.NIFTI1:qf(e,t)&&(r=new ur.NIFTI2),r?r.readHeader(e):console.error("That file does not appear to be NIFTI!"),r}ae.readHeader=HT;function WT(e){return e.extensionFlag[0]!=0}ae.hasExtension=WT;function XT(e,t){var r=e.vox_offset,n=1,i=1;e.dims[4]&&(n=e.dims[4]),e.dims[5]&&(i=e.dims[5]);var s=e.dims[1]*e.dims[2]*e.dims[3]*n*i*(e.numBitsPerVoxel/8);return t.slice(r,r+s)}ae.readImage=XT;function ZT(e,t){var r=e.getExtensionLocation(),n=e.extensionSize;return t.slice(r,r+n)}ae.readExtension=ZT;function QT(e,t){var r=e.getExtensionLocation(),n=e.extensionSize;return t.slice(r+8,r+n)}ae.readExtensionData=QT});var CS=q(Ah=>{var _S;(function(e){typeof DO_NOT_EXPORT_CRC>"u"?typeof Ah=="object"?e(Ah):typeof define=="function"&&define.amd?define(function(){var t={};return e(t),t}):e(_S={}):e(_S={})})(function(e){e.version="1.2.2";function t(){for(var v=0,w=new Array(256),x=0;x!=256;++x)v=x,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,v=v&1?-2097792136^v>>>1:v>>>1,w[x]=v;return typeof Int32Array<"u"?new Int32Array(w):w}var r=t();function n(v){var w=0,x=0,T=0,M=typeof Int32Array<"u"?new Int32Array(4096):new Array(4096);for(T=0;T!=256;++T)M[T]=v[T];for(T=0;T!=256;++T)for(x=v[T],w=256+T;w<4096;w+=256)x=M[w]=x>>>8^v[x&255];var P=[];for(T=1;T!=16;++T)P[T-1]=typeof Int32Array<"u"?M.subarray(T*256,T*256+256):M.slice(T*256,T*256+256);return P}var i=n(r),s=i[0],o=i[1],a=i[2],c=i[3],u=i[4],l=i[5],f=i[6],h=i[7],p=i[8],d=i[9],m=i[10],g=i[11],y=i[12],I=i[13],_=i[14];function E(v,w){for(var x=w^-1,T=0,M=v.length;T<M;)x=x>>>8^r[(x^v.charCodeAt(T++))&255];return~x}function b(v,w){for(var x=w^-1,T=v.length-15,M=0;M<T;)x=_[v[M++]^x&255]^I[v[M++]^x>>8&255]^y[v[M++]^x>>16&255]^g[v[M++]^x>>>24]^m[v[M++]]^d[v[M++]]^p[v[M++]]^h[v[M++]]^f[v[M++]]^l[v[M++]]^u[v[M++]]^c[v[M++]]^a[v[M++]]^o[v[M++]]^s[v[M++]]^r[v[M++]];for(T+=15;M<T;)x=x>>>8^r[(x^v[M++])&255];return~x}function C(v,w){for(var x=w^-1,T=0,M=v.length,P=0,F=0;T<M;)P=v.charCodeAt(T++),P<128?x=x>>>8^r[(x^P)&255]:P<2048?(x=x>>>8^r[(x^(192|P>>6&31))&255],x=x>>>8^r[(x^(128|P&63))&255]):P>=55296&&P<57344?(P=(P&1023)+64,F=v.charCodeAt(T++)&1023,x=x>>>8^r[(x^(240|P>>8&7))&255],x=x>>>8^r[(x^(128|P>>2&63))&255],x=x>>>8^r[(x^(128|F>>6&15|(P&3)<<4))&255],x=x>>>8^r[(x^(128|F&63))&255]):(x=x>>>8^r[(x^(224|P>>12&15))&255],x=x>>>8^r[(x^(128|P>>6&63))&255],x=x>>>8^r[(x^(128|P&63))&255]);return~x}e.table=r,e.bstr=E,e.buf=b,e.str=C})});var yw=q(Gh=>{var gw;(function(e){typeof DO_NOT_EXPORT_CRC>"u"?typeof Gh=="object"?e(Gh):typeof define=="function"&&define.amd?define(function(){var t={};return e(t),t}):e(gw={}):e(gw={})})(function(e){e.version="1.2.2";function t(){for(var v=0,w=new Array(256),x=0;x!=256;++x)v=x,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,v=v&1?-306674912^v>>>1:v>>>1,w[x]=v;return typeof Int32Array<"u"?new Int32Array(w):w}var r=t();function n(v){var w=0,x=0,T=0,M=typeof Int32Array<"u"?new Int32Array(4096):new Array(4096);for(T=0;T!=256;++T)M[T]=v[T];for(T=0;T!=256;++T)for(x=v[T],w=256+T;w<4096;w+=256)x=M[w]=x>>>8^v[x&255];var P=[];for(T=1;T!=16;++T)P[T-1]=typeof Int32Array<"u"?M.subarray(T*256,T*256+256):M.slice(T*256,T*256+256);return P}var i=n(r),s=i[0],o=i[1],a=i[2],c=i[3],u=i[4],l=i[5],f=i[6],h=i[7],p=i[8],d=i[9],m=i[10],g=i[11],y=i[12],I=i[13],_=i[14];function E(v,w){for(var x=w^-1,T=0,M=v.length;T<M;)x=x>>>8^r[(x^v.charCodeAt(T++))&255];return~x}function b(v,w){for(var x=w^-1,T=v.length-15,M=0;M<T;)x=_[v[M++]^x&255]^I[v[M++]^x>>8&255]^y[v[M++]^x>>16&255]^g[v[M++]^x>>>24]^m[v[M++]]^d[v[M++]]^p[v[M++]]^h[v[M++]]^f[v[M++]]^l[v[M++]]^u[v[M++]]^c[v[M++]]^a[v[M++]]^o[v[M++]]^s[v[M++]]^r[v[M++]];for(T+=15;M<T;)x=x>>>8^r[(x^v[M++])&255];return~x}function C(v,w){for(var x=w^-1,T=0,M=v.length,P=0,F=0;T<M;)P=v.charCodeAt(T++),P<128?x=x>>>8^r[(x^P)&255]:P<2048?(x=x>>>8^r[(x^(192|P>>6&31))&255],x=x>>>8^r[(x^(128|P&63))&255]):P>=55296&&P<57344?(P=(P&1023)+64,F=v.charCodeAt(T++)&1023,x=x>>>8^r[(x^(240|P>>8&7))&255],x=x>>>8^r[(x^(128|P>>2&63))&255],x=x>>>8^r[(x^(128|F>>6&15|(P&3)<<4))&255],x=x>>>8^r[(x^(128|F&63))&255]):(x=x>>>8^r[(x^(224|P>>12&15))&255],x=x>>>8^r[(x^(128|P>>6&63))&255],x=x>>>8^r[(x^(128|P&63))&255]);return~x}e.table=r,e.bstr=E,e.buf=b,e.str=C})});var yP=Ri(Ld(),1),vP=Ri(jd(),1);var Rb=typeof global=="object"&&global&&global.Object===Object&&global,qd=Rb;var Ob=typeof self=="object"&&self&&self.Object===Object&&self,Ub=qd||Ob||Function("return this")(),Js=Ub;var Lb=Js.Symbol,Cn=Lb;var Kd=Object.prototype,Fb=Kd.hasOwnProperty,Bb=Kd.toString,zi=Cn?Cn.toStringTag:void 0;function zb(e){var t=Fb.call(e,zi),r=e[zi];try{e[zi]=void 0;var n=!0}catch{}var i=Bb.call(e);return n&&(t?e[zi]=r:delete e[zi]),i}var Yd=zb;var $b=Object.prototype,Vb=$b.toString;function Gb(e){return Vb.call(e)}var Jd=Gb;var jb="[object Null]",qb="[object Undefined]",Hd=Cn?Cn.toStringTag:void 0;function Kb(e){return e==null?e===void 0?qb:jb:Hd&&Hd in Object(e)?Yd(e):Jd(e)}var Wd=Kb;function Yb(e){return e!=null&&typeof e=="object"}var Xd=Yb;var Jb="[object Symbol]";function Hb(e){return typeof e=="symbol"||Xd(e)&&Wd(e)==Jb}var Zd=Hb;var Wb=/\s/;function Xb(e){for(var t=e.length;t--&&Wb.test(e.charAt(t)););return t}var Qd=Xb;var Zb=/^\s+/;function Qb(e){return e&&e.slice(0,Qd(e)+1).replace(Zb,"")}var em=Qb;function eI(e){var t=typeof e;return e!=null&&(t=="object"||t=="function")}var en=eI;var tm=NaN,tI=/^[-+]0x[0-9a-f]+$/i,rI=/^0b[01]+$/i,nI=/^0o[0-7]+$/i,iI=parseInt;function sI(e){if(typeof e=="number")return e;if(Zd(e))return tm;if(en(e)){var t=typeof e.valueOf=="function"?e.valueOf():e;e=en(t)?t+"":t}if(typeof e!="string")return e===0?e:+e;e=em(e);var r=rI.test(e);return r||nI.test(e)?iI(e.slice(2),r?2:8):tI.test(e)?tm:+e}var Su=sI;var oI=function(){return Js.Date.now()},Hs=oI;var aI="Expected a function",cI=Math.max,uI=Math.min;function lI(e,t,r){var n,i,s,o,a,c,u=0,l=!1,f=!1,h=!0;if(typeof e!="function")throw new TypeError(aI);t=Su(t)||0,en(r)&&(l=!!r.leading,f="maxWait"in r,s=f?cI(Su(r.maxWait)||0,t):s,h="trailing"in r?!!r.trailing:h);function p(C){var v=n,w=i;return n=i=void 0,u=C,o=e.apply(w,v),o}function d(C){return u=C,a=setTimeout(y,t),l?p(C):o}function m(C){var v=C-c,w=C-u,x=t-v;return f?uI(x,s-w):x}function g(C){var v=C-c,w=C-u;return c===void 0||v>=t||v<0||f&&w>=s}function y(){var C=Hs();if(g(C))return I(C);a=setTimeout(y,m(C))}function I(C){return a=void 0,h&&n?p(C):(n=i=void 0,o)}function _(){a!==void 0&&clearTimeout(a),u=0,n=c=i=a=void 0}function E(){return a===void 0?o:I(Hs())}function b(){var C=Hs(),v=g(C);if(n=arguments,i=this,c=C,v){if(a===void 0)return d(c);if(f)return clearTimeout(a),a=setTimeout(y,t),p(c)}return a===void 0&&(a=setTimeout(y,t)),o}return b.cancel=_,b.flush=E,b}var tn=lI;var fI="Expected a function";function hI(e,t,r){var n=!0,i=!0;if(typeof e!="function")throw new TypeError(fI);return en(r)&&(n="leading"in r?!!r.leading:n,i="trailing"in r?!!r.trailing:i),tn(e,t,{leading:n,maxWait:t,trailing:i})}var Ws=hI;function pI(e){typeof e=="object"?e.dispose():e()}function wu(e){for(let t=e.length;t>0;--t)pI(e[t-1])}function dI(e,t,r,n){return e.addEventListener(t,r,n),()=>e.removeEventListener(t,r,n)}var be=class{refCount=1;wasDisposed;disposers;addRef(){return++this.refCount,this}disposedStacks;dispose(){--this.refCount===0&&this.refCountReachedZero()}[Symbol.dispose](){this.dispose()}refCountReachedZero(){this.disposed();let{disposers:t}=this;t!==void 0&&(wu(t),this.disposers=void 0),this.wasDisposed=!0}disposed(){}registerDisposer(t){let{disposers:r}=this;return r==null?this.disposers=[t]:r.push(t),t}unregisterDisposer(t){let{disposers:r}=this;if(r!=null){let n=r.indexOf(t);n!==-1&&r.splice(n,1)}return t}registerEventListener(t,r,n,i){this.registerDisposer(dI(t,r,n,i))}registerCancellable(t){return this.registerDisposer(()=>{t.cancel()}),t}},Xs=class extends be{constructor(t){super(),this.value=t}};var Gt=class{handlers=new Set;count=0;constructor(){let t=this;this.dispatch=function(){++t.count,t.handlers.forEach(r=>{r.apply(this,arguments)})}}add(t){return this.handlers.add(t),()=>this.remove(t)}addOnce(t){let{handlers:r}=this;function n(...i){r.delete(n),t(...i)}r.add(n)}remove(t){return this.handlers.delete(t)}dispatch;dispose(){this.handlers=void 0}};var $e=class extends Gt{};var jt=class{constructor(t){this.value_=t}get value(){return this.value_}set value(t){t!==this.value_&&(this.value_=t,this.changed.dispatch())}changed=new $e};function rm(e,...t){let r=t.map(c=>c.value),n=t.length,i=new be,s=e(i,...r),o=tn(()=>{let c=!1;for(let u=0;u<n;++u){let f=t[u].value;r[u]!==f&&(r[u]=f,c=!0)}c&&(i.dispose(),i=new be,s=e(i,...r))},0),a=t.map(c=>c.changed.add(o));return{flush(){o.flush()},dispose(){o.cancel(),wu(a),i.dispose()},get value(){return o.flush(),s}}}function nm(e,t){if(e===void 0)return;if(e.aborted){t(e.reason);return}function r(){t(this.reason)}return e.addEventListener("abort",r,{once:!0}),{[Symbol.dispose](){e.removeEventListener("abort",r)}}}var Zs=class{consumers=new Map;controller=new AbortController;retainCount=0;get signal(){return this.controller.signal}addConsumer(t){if(!this.controller.signal.aborted){if(t!==void 0){let n=function(){i.consumers.delete(n),--i.retainCount===0&&(i.controller.abort(),i[Symbol.dispose]())};var r=n;if(t.aborted)return;let i=this;t.addEventListener("abort",n,{once:!0})}++this.retainCount}}[Symbol.dispose](){for(let[t,r]of this.consumers)r.removeEventListener("abort",t);this.consumers.clear(),this.retainCount=0}start(){this.retainCount===0&&this.controller.abort()}};function im(e,t){let{promise:r,resolve:n,reject:i}=Promise.withResolvers(),s=nm(e,t);return{promise:r,resolve:o=>{s?.[Symbol.dispose](),n(o)},reject:o=>{s?.[Symbol.dispose](),i(o)}}}function Qs(e,t){return t===void 0?e:t.aborted?Promise.reject(t.reason):new Promise((r,n)=>{let i=nm(t,s=>{n(s)});e.then(s=>{i?.[Symbol.dispose](),r(s)},s=>{i?.[Symbol.dispose](),n(s)})})}var He=class{constructor(t,r){this.listener=t;let{id:n=Math.random(),startTime:i=Date.now(),message:s}=r;this.id=n,this.startTime=i,this.message=s,t.addSpan(this)}id;startTime;message;[Symbol.dispose](){this.listener.removeSpan(this.id)}},Eu=class{items=new Map;add(t){let{items:r}=this,n=(r.get(t)??0)+1;return r.set(t,n),n}delete(t){let{items:r}=this,n=r.get(t);return n>1?(n-=1,r.set(t,n),n):(r.delete(t),0)}has(t){return this.items.has(t)}keys(){return this.items.keys()}entries(){return this.items.entries()}[Symbol.iterator](){return this.items.keys()}},bu=class{constructor(t){this.getKey=t}items=new Map;add(t){let{items:r}=this,n=this.getKey(t),i=r.get(n);return i===void 0?(r.set(n,{value:t,count:1}),1):i.count+=1}delete(t){return this.deleteKey(this.getKey(t))}deleteKey(t){let{items:r}=this,n=r.get(t);return n!==void 0&&n.count>1?n.count-=1:(r.delete(t),0)}has(t){return this.items.has(this.getKey(t))}*[Symbol.iterator](){for(let t of this.items.values())yield t.value}};function mI(e){return e.id}var Iu=class extends bu{constructor(){super(mI)}},eo=class{spans=new Iu;listeners=new Eu;addSpan(t){if(this.spans.add(t)===1)for(let r of this.listeners)r.addSpan(t)}removeSpan(t){if(this.spans.deleteKey(t)===0)for(let r of this.listeners)r.removeSpan(t)}addListener(t){if(t!==void 0&&this.listeners.add(t)===1)for(let r of this.spans)t.addSpan(r)}removeListener(t){if(t!==void 0&&this.listeners.delete(t)===0)for(let r of this.spans)t.removeSpan(r.id)}};var ro=!(typeof Window<"u"&&self instanceof Window),om=!1,sm=!1,_u="rpc.promise.response",am="rpc.promise.cancel",cm="rpc.promise.addProgressSpan",um="rpc.promise.removeProgressSpan",lm="rpc.ready",Cu=new Map;function X(e,t){Cu.set(e,t)}var Au=class{constructor(t,r){this.rpc=t,this.id=r}addSpan(t){this.rpc.invoke(cm,{id:this.id,span:{id:t.id,message:t.message,startTime:t.startTime}})}removeSpan(t){this.rpc.invoke(um,{id:this.id,spanId:t})}};function xt(e,t){X(e,function(r){let n=r.id,i=new AbortController,s;r.progressListener===!0&&(s=new Au(this,n));let o=t.call(this,r,{signal:i.signal,progressListener:s});this.set(n,{promise:o,abortController:i}),o.then(({value:a,transfers:c})=>{this.delete(n),this.invoke(_u,{id:n,value:a},c)},a=>{this.delete(n),this.invoke(_u,{id:n,error:a})})})}X(am,function(e){let t=e.id,r=this.get(t);if(r!==void 0){let{abortController:n}=r;n.abort()}});X(_u,function(e){let t=e.id,{resolve:r,reject:n}=this.get(t);this.delete(t),Object.prototype.hasOwnProperty.call(e,"value")?r(e.value):n(e.error)});X(cm,function(e){let t=e.id,{progressListener:r}=this.get(t);new He(r,e.span)});X(um,function(e){let t=e.id,{progressListener:r}=this.get(t);r.removeSpan(e.spanId)});X(lm,function(e){this.onPeerReady()});var gI=ro?-1:0,to=class{constructor(t,r){this.target=t,r&&(this.queue=[]),t.onmessage=n=>{let i=n.data;if(sm&&console.log("Received message",i),Cu.get(i.functionName)===void 0)throw new Error(`Missing RPC function: ${i.functionName}`);Cu.get(i.functionName).call(this,i)}}objects=new Map;nextId=gI;queue;sendReady(){this.invoke(lm,{})}onPeerReady(){let{queue:t}=this;if(t!==void 0){this.queue=void 0;for(let{data:r,transfers:n}of t)this.target.postMessage(r,n)}}get numObjects(){return this.objects.size}set(t,r){this.objects.set(t,r)}delete(t){this.objects.delete(t)}get(t){return this.objects.get(t)}getRef(t){let r=t.id,n=this.get(r);return n.referencedGeneration=t.gen,n.addRef(),n}getOptionalRef(t){if(t===void 0)return;let r=t.id,n=this.get(r);return n.referencedGeneration=t.gen,n.addRef(),n}invoke(t,r,n){r.functionName=t,sm&&console.trace("Sending message",r);let{queue:i}=this;if(i!==void 0){i.push({data:r,transfers:n});return}this.target.postMessage(r,n)}promiseInvoke(t,r,n){let i,s,o;if(n!==void 0&&({signal:i,progressListener:s,transfers:o}=n),i?.aborted)return Promise.reject(i.reason);s!==void 0&&(r.progressListener=!0);let a=r.id=this.newId();this.invoke(t,r,o);let{promise:c,resolve:u,reject:l}=i===void 0?Promise.withResolvers():im(i,()=>{this.invoke(am,{id:a})});return this.set(a,{resolve:u,reject:l,progressListener:s}),c}newId(){return ro?this.nextId--:this.nextId++}},rn=class extends be{rpc=null;rpcId=null;isOwner;unreferencedGeneration;referencedGeneration;initializeSharedObject(t,r=t.newId()){this.rpc=t,this.rpcId=r,this.isOwner=!1,t.set(r,this)}initializeCounterpart(t,r={}){this.initializeSharedObject(t),this.unreferencedGeneration=0,this.referencedGeneration=0,this.isOwner=!0,r.id=this.rpcId,r.type=this.RPC_TYPE_ID,t.invoke("SharedObject.new",r)}dispose(){super.dispose()}addCounterpartRef(){return{id:this.rpcId,gen:++this.referencedGeneration}}refCountReachedZero(){this.isOwner===!0?this.referencedGeneration===this.unreferencedGeneration&&this.ownerDispose():this.isOwner===!1?this.rpc.invoke("SharedObject.refCountReachedZero",{id:this.rpcId,gen:this.referencedGeneration}):super.refCountReachedZero()}ownerDispose(){om&&console.log(`[${ro}] #rpc object = ${this.rpc.numObjects}`);let{rpc:t,rpcId:r}=this;super.refCountReachedZero(),t.delete(r),t.invoke("SharedObject.dispose",{id:r})}counterpartRefCountReachedZero(t){this.unreferencedGeneration=t,this.refCount===0&&t===this.referencedGeneration&&this.ownerDispose()}};function Mu(e,t,r={}){t!=null&&e.initializeSharedObject(t,r.id)}var me=class extends rn{constructor(t,r={}){super(),Mu(this,t,r)}};X("SharedObject.dispose",function(e){let t=this.get(e.id);if(t.refCount!==0)throw new Error("Attempted to dispose object with non-zero reference count.");om&&console.log(`[${ro}] #rpc objects: ${this.numObjects}`),t.disposed(),this.delete(t.rpcId),t.rpcId=null,t.rpc=null});X("SharedObject.refCountReachedZero",function(e){let t=this.get(e.id),r=e.gen;t.counterpartRefCountReachedZero(r)});var fm=new Map;function hm(e){return t=>{t.prototype.RPC_TYPE_ID=e}}function G(e){return t=>{if(e!==void 0)t.prototype.RPC_TYPE_ID=e;else if(e=t.prototype.RPC_TYPE_ID,e===void 0)throw new Error("RPC_TYPE_ID should have already been defined");fm.set(e,t)}}X("SharedObject.new",function(e){let t=this,r=e.type,n=fm.get(r),i=new n(t,e);--i.refCount});var yI=Object.defineProperty,vI=Object.getOwnPropertyDescriptor,xI=(e,t,r,n)=>{for(var i=n>1?void 0:n?vI(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&yI(t,r,i),i};var pm="SharedWatchableValue.changed",no=class extends me{base;updatingValue_=!1;constructor(e,t={}){super(e,t),e!==void 0&&(this.base=new jt(t.value),this.setupChangedHandler())}initializeCounterpart(e,t={}){t.value=this.value,super.initializeCounterpart(e,t)}setupChangedHandler(){this.registerDisposer(this.base.changed.add(()=>{if(this.updatingValue_)this.updatingValue_=!1;else{let{rpc:e}=this;e!==null&&e.invoke(pm,{id:this.rpcId,value:this.value})}}))}static makeFromExisting(e,t){let r=new no;return r.base=t,r.setupChangedHandler(),r.initializeCounterpart(e),r}static make(e,t){return no.makeFromExisting(e,new jt(t))}get value(){return this.base.value}set value(e){this.base.value=e}get changed(){return this.base.changed}};no=xI([G("SharedWatchableValue")],no);X(pm,function(e){let t=this.get(e.id);t.updatingValue_=!0,t.base.value=e.value,t.updatingValue_=!1});var z=(e=>(e[e.GPU_MEMORY=0]="GPU_MEMORY",e[e.SYSTEM_MEMORY=1]="SYSTEM_MEMORY",e[e.SYSTEM_MEMORY_WORKER=2]="SYSTEM_MEMORY_WORKER",e[e.DOWNLOADING=3]="DOWNLOADING",e[e.QUEUED=4]="QUEUED",e[e.NEW=5]="NEW",e[e.FAILED=6]="FAILED",e[e.EXPIRED=7]="EXPIRED",e))(z||{}),dm=8,se=(e=>(e[e.FIRST_TIER=0]="FIRST_TIER",e[e.FIRST_ORDERED_TIER=0]="FIRST_ORDERED_TIER",e[e.VISIBLE=0]="VISIBLE",e[e.PREFETCH=1]="PREFETCH",e[e.LAST_ORDERED_TIER=1]="LAST_ORDERED_TIER",e[e.RECENT=2]="RECENT",e[e.LAST_TIER=2]="LAST_TIER",e))(se||{}),Tu=3,io=(e=>(e[e.totalTime=0]="totalTime",e[e.totalChunks=1]="totalChunks",e))(io||{}),$i=(e=>(e[e.numChunks=0]="numChunks",e[e.systemMemoryBytes=1]="systemMemoryBytes",e[e.gpuMemoryBytes=2]="gpuMemoryBytes",e))($i||{}),An=3,SI=2,mm=dm*Tu*An+SI;function gm(e,t){return e*Tu+t}function ku(e){return dm*Tu*An+e}var ym=1e13,vm="ChunkQueueManager",xm="ChunkManager",Sm="ChunkSource.invalidate",wm="ChunkQueueManager.requestChunkStatistics",Em="ChunkManager.chunkLayerStatistics";function so(e){let{next:t,prev:r}=e;return{insertAfter(n,i){let s=n[t];i[t]=s,i[r]=n,n[t]=i,s[r]=i},insertBefore(n,i){let s=n[r];i[r]=s,i[t]=n,n[r]=i,s[t]=i},front(n){let i=n[t];return i===n?null:i},back(n){let i=n[r];return i===n?null:i},pop(n){let i=n[t],s=n[r];return i[r]=s,s[t]=i,n[t]=null,n[r]=null,n},*iterator(n){for(let i=n[t];i!==n;i=i[t])yield i},*reverseIterator(n){for(let i=n[r];i!==n;i=i[r])yield i},initializeHead(n){n[t]=n[r]=n}}}function mr(e,t){return e<t?-1:e>t?1:0}function wI(e,t){return BigInt(e)|BigInt(t)<<32n}function Nu(){let e=Math.random()*4294967296>>>0,t=Math.random()*4294967296>>>0;return wI(e,t)}var Du=0xffffffffffffffffn;var Z=1e-6,Se=typeof Float32Array<"u"?Float32Array:Array,Mt=Math.random;var wR=Math.PI/180;Math.hypot||(Math.hypot=function(){for(var e=0,t=arguments.length;t--;)e+=arguments[t]*arguments[t];return Math.sqrt(e)});var Tt={};Pi(Tt,{add:()=>jI,adjoint:()=>kI,clone:()=>bI,copy:()=>II,create:()=>Pu,determinant:()=>NI,equals:()=>JI,exactEquals:()=>YI,frob:()=>GI,fromMat2d:()=>FI,fromMat4:()=>EI,fromQuat:()=>BI,fromRotation:()=>UI,fromScaling:()=>LI,fromTranslation:()=>OI,fromValues:()=>_I,identity:()=>AI,invert:()=>TI,mul:()=>HI,multiply:()=>bm,multiplyScalar:()=>qI,multiplyScalarAndAdd:()=>KI,normalFromMat4:()=>zI,projection:()=>$I,rotate:()=>PI,scale:()=>RI,set:()=>CI,str:()=>VI,sub:()=>WI,subtract:()=>Im,translate:()=>DI,transpose:()=>MI});function Pu(){var e=new Se(9);return Se!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[5]=0,e[6]=0,e[7]=0),e[0]=1,e[4]=1,e[8]=1,e}function EI(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[4],e[4]=t[5],e[5]=t[6],e[6]=t[8],e[7]=t[9],e[8]=t[10],e}function bI(e){var t=new Se(9);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t[4]=e[4],t[5]=e[5],t[6]=e[6],t[7]=e[7],t[8]=e[8],t}function II(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[8]=t[8],e}function _I(e,t,r,n,i,s,o,a,c){var u=new Se(9);return u[0]=e,u[1]=t,u[2]=r,u[3]=n,u[4]=i,u[5]=s,u[6]=o,u[7]=a,u[8]=c,u}function CI(e,t,r,n,i,s,o,a,c,u){return e[0]=t,e[1]=r,e[2]=n,e[3]=i,e[4]=s,e[5]=o,e[6]=a,e[7]=c,e[8]=u,e}function AI(e){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=1,e[5]=0,e[6]=0,e[7]=0,e[8]=1,e}function MI(e,t){if(e===t){var r=t[1],n=t[2],i=t[5];e[1]=t[3],e[2]=t[6],e[3]=r,e[5]=t[7],e[6]=n,e[7]=i}else e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8];return e}function TI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],u=t[7],l=t[8],f=l*o-a*u,h=-l*s+a*c,p=u*s-o*c,d=r*f+n*h+i*p;return d?(d=1/d,e[0]=f*d,e[1]=(-l*n+i*u)*d,e[2]=(a*n-i*o)*d,e[3]=h*d,e[4]=(l*r-i*c)*d,e[5]=(-a*r+i*s)*d,e[6]=p*d,e[7]=(-u*r+n*c)*d,e[8]=(o*r-n*s)*d,e):null}function kI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],u=t[7],l=t[8];return e[0]=o*l-a*u,e[1]=i*u-n*l,e[2]=n*a-i*o,e[3]=a*c-s*l,e[4]=r*l-i*c,e[5]=i*s-r*a,e[6]=s*u-o*c,e[7]=n*c-r*u,e[8]=r*o-n*s,e}function NI(e){var t=e[0],r=e[1],n=e[2],i=e[3],s=e[4],o=e[5],a=e[6],c=e[7],u=e[8];return t*(u*s-o*c)+r*(-u*i+o*a)+n*(c*i-s*a)}function bm(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],u=t[6],l=t[7],f=t[8],h=r[0],p=r[1],d=r[2],m=r[3],g=r[4],y=r[5],I=r[6],_=r[7],E=r[8];return e[0]=h*n+p*o+d*u,e[1]=h*i+p*a+d*l,e[2]=h*s+p*c+d*f,e[3]=m*n+g*o+y*u,e[4]=m*i+g*a+y*l,e[5]=m*s+g*c+y*f,e[6]=I*n+_*o+E*u,e[7]=I*i+_*a+E*l,e[8]=I*s+_*c+E*f,e}function DI(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],u=t[6],l=t[7],f=t[8],h=r[0],p=r[1];return e[0]=n,e[1]=i,e[2]=s,e[3]=o,e[4]=a,e[5]=c,e[6]=h*n+p*o+u,e[7]=h*i+p*a+l,e[8]=h*s+p*c+f,e}function PI(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],u=t[6],l=t[7],f=t[8],h=Math.sin(r),p=Math.cos(r);return e[0]=p*n+h*o,e[1]=p*i+h*a,e[2]=p*s+h*c,e[3]=p*o-h*n,e[4]=p*a-h*i,e[5]=p*c-h*s,e[6]=u,e[7]=l,e[8]=f,e}function RI(e,t,r){var n=r[0],i=r[1];return e[0]=n*t[0],e[1]=n*t[1],e[2]=n*t[2],e[3]=i*t[3],e[4]=i*t[4],e[5]=i*t[5],e[6]=t[6],e[7]=t[7],e[8]=t[8],e}function OI(e,t){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=1,e[5]=0,e[6]=t[0],e[7]=t[1],e[8]=1,e}function UI(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=n,e[1]=r,e[2]=0,e[3]=-r,e[4]=n,e[5]=0,e[6]=0,e[7]=0,e[8]=1,e}function LI(e,t){return e[0]=t[0],e[1]=0,e[2]=0,e[3]=0,e[4]=t[1],e[5]=0,e[6]=0,e[7]=0,e[8]=1,e}function FI(e,t){return e[0]=t[0],e[1]=t[1],e[2]=0,e[3]=t[2],e[4]=t[3],e[5]=0,e[6]=t[4],e[7]=t[5],e[8]=1,e}function BI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r+r,a=n+n,c=i+i,u=r*o,l=n*o,f=n*a,h=i*o,p=i*a,d=i*c,m=s*o,g=s*a,y=s*c;return e[0]=1-f-d,e[3]=l-y,e[6]=h+g,e[1]=l+y,e[4]=1-u-d,e[7]=p-m,e[2]=h-g,e[5]=p+m,e[8]=1-u-f,e}function zI(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],u=t[7],l=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15],I=r*a-n*o,_=r*c-i*o,E=r*u-s*o,b=n*c-i*a,C=n*u-s*a,v=i*u-s*c,w=l*m-f*d,x=l*g-h*d,T=l*y-p*d,M=f*g-h*m,P=f*y-p*m,F=h*y-p*g,S=I*F-_*P+E*M+b*T-C*x+v*w;return S?(S=1/S,e[0]=(a*F-c*P+u*M)*S,e[1]=(c*T-o*F-u*x)*S,e[2]=(o*P-a*T+u*w)*S,e[3]=(i*P-n*F-s*M)*S,e[4]=(r*F-i*T+s*x)*S,e[5]=(n*T-r*P-s*w)*S,e[6]=(m*v-g*C+y*b)*S,e[7]=(g*E-d*v-y*_)*S,e[8]=(d*C-m*E+y*I)*S,e):null}function $I(e,t,r){return e[0]=2/t,e[1]=0,e[2]=0,e[3]=0,e[4]=-2/r,e[5]=0,e[6]=-1,e[7]=1,e[8]=1,e}function VI(e){return"mat3("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+", "+e[4]+", "+e[5]+", "+e[6]+", "+e[7]+", "+e[8]+")"}function GI(e){return Math.hypot(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8])}function jI(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e[3]=t[3]+r[3],e[4]=t[4]+r[4],e[5]=t[5]+r[5],e[6]=t[6]+r[6],e[7]=t[7]+r[7],e[8]=t[8]+r[8],e}function Im(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e[3]=t[3]-r[3],e[4]=t[4]-r[4],e[5]=t[5]-r[5],e[6]=t[6]-r[6],e[7]=t[7]-r[7],e[8]=t[8]-r[8],e}function qI(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e[3]=t[3]*r,e[4]=t[4]*r,e[5]=t[5]*r,e[6]=t[6]*r,e[7]=t[7]*r,e[8]=t[8]*r,e}function KI(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e[3]=t[3]+r[3]*n,e[4]=t[4]+r[4]*n,e[5]=t[5]+r[5]*n,e[6]=t[6]+r[6]*n,e[7]=t[7]+r[7]*n,e[8]=t[8]+r[8]*n,e}function YI(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]&&e[3]===t[3]&&e[4]===t[4]&&e[5]===t[5]&&e[6]===t[6]&&e[7]===t[7]&&e[8]===t[8]}function JI(e,t){var r=e[0],n=e[1],i=e[2],s=e[3],o=e[4],a=e[5],c=e[6],u=e[7],l=e[8],f=t[0],h=t[1],p=t[2],d=t[3],m=t[4],g=t[5],y=t[6],I=t[7],_=t[8];return Math.abs(r-f)<=Z*Math.max(1,Math.abs(r),Math.abs(f))&&Math.abs(n-h)<=Z*Math.max(1,Math.abs(n),Math.abs(h))&&Math.abs(i-p)<=Z*Math.max(1,Math.abs(i),Math.abs(p))&&Math.abs(s-d)<=Z*Math.max(1,Math.abs(s),Math.abs(d))&&Math.abs(o-m)<=Z*Math.max(1,Math.abs(o),Math.abs(m))&&Math.abs(a-g)<=Z*Math.max(1,Math.abs(a),Math.abs(g))&&Math.abs(c-y)<=Z*Math.max(1,Math.abs(c),Math.abs(y))&&Math.abs(u-I)<=Z*Math.max(1,Math.abs(u),Math.abs(I))&&Math.abs(l-_)<=Z*Math.max(1,Math.abs(l),Math.abs(_))}var HI=bm,WI=Im;var ge={};Pi(ge,{add:()=>D2,adjoint:()=>i2,clone:()=>ZI,copy:()=>QI,create:()=>XI,determinant:()=>s2,equals:()=>U2,exactEquals:()=>O2,frob:()=>N2,fromQuat:()=>b2,fromQuat2:()=>v2,fromRotation:()=>d2,fromRotationTranslation:()=>Am,fromRotationTranslationScale:()=>w2,fromRotationTranslationScaleOrigin:()=>E2,fromScaling:()=>p2,fromTranslation:()=>h2,fromValues:()=>e2,fromXRotation:()=>m2,fromYRotation:()=>g2,fromZRotation:()=>y2,frustum:()=>I2,getRotation:()=>S2,getScaling:()=>Mm,getTranslation:()=>x2,identity:()=>_m,invert:()=>n2,lookAt:()=>M2,mul:()=>L2,multiply:()=>Cm,multiplyScalar:()=>P2,multiplyScalarAndAdd:()=>R2,ortho:()=>A2,perspective:()=>_2,perspectiveFromFieldOfView:()=>C2,rotate:()=>c2,rotateX:()=>u2,rotateY:()=>l2,rotateZ:()=>f2,scale:()=>a2,set:()=>t2,str:()=>k2,sub:()=>F2,subtract:()=>Tm,targetTo:()=>T2,translate:()=>o2,transpose:()=>r2});function XI(){var e=new Se(16);return Se!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0),e[0]=1,e[5]=1,e[10]=1,e[15]=1,e}function ZI(e){var t=new Se(16);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t[4]=e[4],t[5]=e[5],t[6]=e[6],t[7]=e[7],t[8]=e[8],t[9]=e[9],t[10]=e[10],t[11]=e[11],t[12]=e[12],t[13]=e[13],t[14]=e[14],t[15]=e[15],t}function QI(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[8]=t[8],e[9]=t[9],e[10]=t[10],e[11]=t[11],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15],e}function e2(e,t,r,n,i,s,o,a,c,u,l,f,h,p,d,m){var g=new Se(16);return g[0]=e,g[1]=t,g[2]=r,g[3]=n,g[4]=i,g[5]=s,g[6]=o,g[7]=a,g[8]=c,g[9]=u,g[10]=l,g[11]=f,g[12]=h,g[13]=p,g[14]=d,g[15]=m,g}function t2(e,t,r,n,i,s,o,a,c,u,l,f,h,p,d,m,g){return e[0]=t,e[1]=r,e[2]=n,e[3]=i,e[4]=s,e[5]=o,e[6]=a,e[7]=c,e[8]=u,e[9]=l,e[10]=f,e[11]=h,e[12]=p,e[13]=d,e[14]=m,e[15]=g,e}function _m(e){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function r2(e,t){if(e===t){var r=t[1],n=t[2],i=t[3],s=t[6],o=t[7],a=t[11];e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=r,e[6]=t[9],e[7]=t[13],e[8]=n,e[9]=s,e[11]=t[14],e[12]=i,e[13]=o,e[14]=a}else e[0]=t[0],e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=t[1],e[5]=t[5],e[6]=t[9],e[7]=t[13],e[8]=t[2],e[9]=t[6],e[10]=t[10],e[11]=t[14],e[12]=t[3],e[13]=t[7],e[14]=t[11],e[15]=t[15];return e}function n2(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],u=t[7],l=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15],I=r*a-n*o,_=r*c-i*o,E=r*u-s*o,b=n*c-i*a,C=n*u-s*a,v=i*u-s*c,w=l*m-f*d,x=l*g-h*d,T=l*y-p*d,M=f*g-h*m,P=f*y-p*m,F=h*y-p*g,S=I*F-_*P+E*M+b*T-C*x+v*w;return S?(S=1/S,e[0]=(a*F-c*P+u*M)*S,e[1]=(i*P-n*F-s*M)*S,e[2]=(m*v-g*C+y*b)*S,e[3]=(h*C-f*v-p*b)*S,e[4]=(c*T-o*F-u*x)*S,e[5]=(r*F-i*T+s*x)*S,e[6]=(g*E-d*v-y*_)*S,e[7]=(l*v-h*E+p*_)*S,e[8]=(o*P-a*T+u*w)*S,e[9]=(n*T-r*P-s*w)*S,e[10]=(d*C-m*E+y*I)*S,e[11]=(f*E-l*C-p*I)*S,e[12]=(a*x-o*M-c*w)*S,e[13]=(r*M-n*x+i*w)*S,e[14]=(m*_-d*b-g*I)*S,e[15]=(l*b-f*_+h*I)*S,e):null}function i2(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],u=t[7],l=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15];return e[0]=a*(h*y-p*g)-f*(c*y-u*g)+m*(c*p-u*h),e[1]=-(n*(h*y-p*g)-f*(i*y-s*g)+m*(i*p-s*h)),e[2]=n*(c*y-u*g)-a*(i*y-s*g)+m*(i*u-s*c),e[3]=-(n*(c*p-u*h)-a*(i*p-s*h)+f*(i*u-s*c)),e[4]=-(o*(h*y-p*g)-l*(c*y-u*g)+d*(c*p-u*h)),e[5]=r*(h*y-p*g)-l*(i*y-s*g)+d*(i*p-s*h),e[6]=-(r*(c*y-u*g)-o*(i*y-s*g)+d*(i*u-s*c)),e[7]=r*(c*p-u*h)-o*(i*p-s*h)+l*(i*u-s*c),e[8]=o*(f*y-p*m)-l*(a*y-u*m)+d*(a*p-u*f),e[9]=-(r*(f*y-p*m)-l*(n*y-s*m)+d*(n*p-s*f)),e[10]=r*(a*y-u*m)-o*(n*y-s*m)+d*(n*u-s*a),e[11]=-(r*(a*p-u*f)-o*(n*p-s*f)+l*(n*u-s*a)),e[12]=-(o*(f*g-h*m)-l*(a*g-c*m)+d*(a*h-c*f)),e[13]=r*(f*g-h*m)-l*(n*g-i*m)+d*(n*h-i*f),e[14]=-(r*(a*g-c*m)-o*(n*g-i*m)+d*(n*c-i*a)),e[15]=r*(a*h-c*f)-o*(n*h-i*f)+l*(n*c-i*a),e}function s2(e){var t=e[0],r=e[1],n=e[2],i=e[3],s=e[4],o=e[5],a=e[6],c=e[7],u=e[8],l=e[9],f=e[10],h=e[11],p=e[12],d=e[13],m=e[14],g=e[15],y=t*o-r*s,I=t*a-n*s,_=t*c-i*s,E=r*a-n*o,b=r*c-i*o,C=n*c-i*a,v=u*d-l*p,w=u*m-f*p,x=u*g-h*p,T=l*m-f*d,M=l*g-h*d,P=f*g-h*m;return y*P-I*M+_*T+E*x-b*w+C*v}function Cm(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=t[4],c=t[5],u=t[6],l=t[7],f=t[8],h=t[9],p=t[10],d=t[11],m=t[12],g=t[13],y=t[14],I=t[15],_=r[0],E=r[1],b=r[2],C=r[3];return e[0]=_*n+E*a+b*f+C*m,e[1]=_*i+E*c+b*h+C*g,e[2]=_*s+E*u+b*p+C*y,e[3]=_*o+E*l+b*d+C*I,_=r[4],E=r[5],b=r[6],C=r[7],e[4]=_*n+E*a+b*f+C*m,e[5]=_*i+E*c+b*h+C*g,e[6]=_*s+E*u+b*p+C*y,e[7]=_*o+E*l+b*d+C*I,_=r[8],E=r[9],b=r[10],C=r[11],e[8]=_*n+E*a+b*f+C*m,e[9]=_*i+E*c+b*h+C*g,e[10]=_*s+E*u+b*p+C*y,e[11]=_*o+E*l+b*d+C*I,_=r[12],E=r[13],b=r[14],C=r[15],e[12]=_*n+E*a+b*f+C*m,e[13]=_*i+E*c+b*h+C*g,e[14]=_*s+E*u+b*p+C*y,e[15]=_*o+E*l+b*d+C*I,e}function o2(e,t,r){var n=r[0],i=r[1],s=r[2],o,a,c,u,l,f,h,p,d,m,g,y;return t===e?(e[12]=t[0]*n+t[4]*i+t[8]*s+t[12],e[13]=t[1]*n+t[5]*i+t[9]*s+t[13],e[14]=t[2]*n+t[6]*i+t[10]*s+t[14],e[15]=t[3]*n+t[7]*i+t[11]*s+t[15]):(o=t[0],a=t[1],c=t[2],u=t[3],l=t[4],f=t[5],h=t[6],p=t[7],d=t[8],m=t[9],g=t[10],y=t[11],e[0]=o,e[1]=a,e[2]=c,e[3]=u,e[4]=l,e[5]=f,e[6]=h,e[7]=p,e[8]=d,e[9]=m,e[10]=g,e[11]=y,e[12]=o*n+l*i+d*s+t[12],e[13]=a*n+f*i+m*s+t[13],e[14]=c*n+h*i+g*s+t[14],e[15]=u*n+p*i+y*s+t[15]),e}function a2(e,t,r){var n=r[0],i=r[1],s=r[2];return e[0]=t[0]*n,e[1]=t[1]*n,e[2]=t[2]*n,e[3]=t[3]*n,e[4]=t[4]*i,e[5]=t[5]*i,e[6]=t[6]*i,e[7]=t[7]*i,e[8]=t[8]*s,e[9]=t[9]*s,e[10]=t[10]*s,e[11]=t[11]*s,e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15],e}function c2(e,t,r,n){var i=n[0],s=n[1],o=n[2],a=Math.hypot(i,s,o),c,u,l,f,h,p,d,m,g,y,I,_,E,b,C,v,w,x,T,M,P,F,S,O;return a<Z?null:(a=1/a,i*=a,s*=a,o*=a,c=Math.sin(r),u=Math.cos(r),l=1-u,f=t[0],h=t[1],p=t[2],d=t[3],m=t[4],g=t[5],y=t[6],I=t[7],_=t[8],E=t[9],b=t[10],C=t[11],v=i*i*l+u,w=s*i*l+o*c,x=o*i*l-s*c,T=i*s*l-o*c,M=s*s*l+u,P=o*s*l+i*c,F=i*o*l+s*c,S=s*o*l-i*c,O=o*o*l+u,e[0]=f*v+m*w+_*x,e[1]=h*v+g*w+E*x,e[2]=p*v+y*w+b*x,e[3]=d*v+I*w+C*x,e[4]=f*T+m*M+_*P,e[5]=h*T+g*M+E*P,e[6]=p*T+y*M+b*P,e[7]=d*T+I*M+C*P,e[8]=f*F+m*S+_*O,e[9]=h*F+g*S+E*O,e[10]=p*F+y*S+b*O,e[11]=d*F+I*S+C*O,t!==e&&(e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e)}function u2(e,t,r){var n=Math.sin(r),i=Math.cos(r),s=t[4],o=t[5],a=t[6],c=t[7],u=t[8],l=t[9],f=t[10],h=t[11];return t!==e&&(e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[4]=s*i+u*n,e[5]=o*i+l*n,e[6]=a*i+f*n,e[7]=c*i+h*n,e[8]=u*i-s*n,e[9]=l*i-o*n,e[10]=f*i-a*n,e[11]=h*i-c*n,e}function l2(e,t,r){var n=Math.sin(r),i=Math.cos(r),s=t[0],o=t[1],a=t[2],c=t[3],u=t[8],l=t[9],f=t[10],h=t[11];return t!==e&&(e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*i-u*n,e[1]=o*i-l*n,e[2]=a*i-f*n,e[3]=c*i-h*n,e[8]=s*n+u*i,e[9]=o*n+l*i,e[10]=a*n+f*i,e[11]=c*n+h*i,e}function f2(e,t,r){var n=Math.sin(r),i=Math.cos(r),s=t[0],o=t[1],a=t[2],c=t[3],u=t[4],l=t[5],f=t[6],h=t[7];return t!==e&&(e[8]=t[8],e[9]=t[9],e[10]=t[10],e[11]=t[11],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*i+u*n,e[1]=o*i+l*n,e[2]=a*i+f*n,e[3]=c*i+h*n,e[4]=u*i-s*n,e[5]=l*i-o*n,e[6]=f*i-a*n,e[7]=h*i-c*n,e}function h2(e,t){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=t[0],e[13]=t[1],e[14]=t[2],e[15]=1,e}function p2(e,t){return e[0]=t[0],e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=t[1],e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=t[2],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function d2(e,t,r){var n=r[0],i=r[1],s=r[2],o=Math.hypot(n,i,s),a,c,u;return o<Z?null:(o=1/o,n*=o,i*=o,s*=o,a=Math.sin(t),c=Math.cos(t),u=1-c,e[0]=n*n*u+c,e[1]=i*n*u+s*a,e[2]=s*n*u-i*a,e[3]=0,e[4]=n*i*u-s*a,e[5]=i*i*u+c,e[6]=s*i*u+n*a,e[7]=0,e[8]=n*s*u+i*a,e[9]=i*s*u-n*a,e[10]=s*s*u+c,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e)}function m2(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n,e[6]=r,e[7]=0,e[8]=0,e[9]=-r,e[10]=n,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function g2(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=n,e[1]=0,e[2]=-r,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=r,e[9]=0,e[10]=n,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function y2(e,t){var r=Math.sin(t),n=Math.cos(t);return e[0]=n,e[1]=r,e[2]=0,e[3]=0,e[4]=-r,e[5]=n,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Am(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=n+n,c=i+i,u=s+s,l=n*a,f=n*c,h=n*u,p=i*c,d=i*u,m=s*u,g=o*a,y=o*c,I=o*u;return e[0]=1-(p+m),e[1]=f+I,e[2]=h-y,e[3]=0,e[4]=f-I,e[5]=1-(l+m),e[6]=d+g,e[7]=0,e[8]=h+y,e[9]=d-g,e[10]=1-(l+p),e[11]=0,e[12]=r[0],e[13]=r[1],e[14]=r[2],e[15]=1,e}function v2(e,t){var r=new Se(3),n=-t[0],i=-t[1],s=-t[2],o=t[3],a=t[4],c=t[5],u=t[6],l=t[7],f=n*n+i*i+s*s+o*o;return f>0?(r[0]=(a*o+l*n+c*s-u*i)*2/f,r[1]=(c*o+l*i+u*n-a*s)*2/f,r[2]=(u*o+l*s+a*i-c*n)*2/f):(r[0]=(a*o+l*n+c*s-u*i)*2,r[1]=(c*o+l*i+u*n-a*s)*2,r[2]=(u*o+l*s+a*i-c*n)*2),Am(e,t,r),e}function x2(e,t){return e[0]=t[12],e[1]=t[13],e[2]=t[14],e}function Mm(e,t){var r=t[0],n=t[1],i=t[2],s=t[4],o=t[5],a=t[6],c=t[8],u=t[9],l=t[10];return e[0]=Math.hypot(r,n,i),e[1]=Math.hypot(s,o,a),e[2]=Math.hypot(c,u,l),e}function S2(e,t){var r=new Se(3);Mm(r,t);var n=1/r[0],i=1/r[1],s=1/r[2],o=t[0]*n,a=t[1]*i,c=t[2]*s,u=t[4]*n,l=t[5]*i,f=t[6]*s,h=t[8]*n,p=t[9]*i,d=t[10]*s,m=o+l+d,g=0;return m>0?(g=Math.sqrt(m+1)*2,e[3]=.25*g,e[0]=(f-p)/g,e[1]=(h-c)/g,e[2]=(a-u)/g):o>l&&o>d?(g=Math.sqrt(1+o-l-d)*2,e[3]=(f-p)/g,e[0]=.25*g,e[1]=(a+u)/g,e[2]=(h+c)/g):l>d?(g=Math.sqrt(1+l-o-d)*2,e[3]=(h-c)/g,e[0]=(a+u)/g,e[1]=.25*g,e[2]=(f+p)/g):(g=Math.sqrt(1+d-o-l)*2,e[3]=(a-u)/g,e[0]=(h+c)/g,e[1]=(f+p)/g,e[2]=.25*g),e}function w2(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=t[3],c=i+i,u=s+s,l=o+o,f=i*c,h=i*u,p=i*l,d=s*u,m=s*l,g=o*l,y=a*c,I=a*u,_=a*l,E=n[0],b=n[1],C=n[2];return e[0]=(1-(d+g))*E,e[1]=(h+_)*E,e[2]=(p-I)*E,e[3]=0,e[4]=(h-_)*b,e[5]=(1-(f+g))*b,e[6]=(m+y)*b,e[7]=0,e[8]=(p+I)*C,e[9]=(m-y)*C,e[10]=(1-(f+d))*C,e[11]=0,e[12]=r[0],e[13]=r[1],e[14]=r[2],e[15]=1,e}function E2(e,t,r,n,i){var s=t[0],o=t[1],a=t[2],c=t[3],u=s+s,l=o+o,f=a+a,h=s*u,p=s*l,d=s*f,m=o*l,g=o*f,y=a*f,I=c*u,_=c*l,E=c*f,b=n[0],C=n[1],v=n[2],w=i[0],x=i[1],T=i[2],M=(1-(m+y))*b,P=(p+E)*b,F=(d-_)*b,S=(p-E)*C,O=(1-(h+y))*C,R=(g+I)*C,N=(d+_)*v,U=(g-I)*v,L=(1-(h+m))*v;return e[0]=M,e[1]=P,e[2]=F,e[3]=0,e[4]=S,e[5]=O,e[6]=R,e[7]=0,e[8]=N,e[9]=U,e[10]=L,e[11]=0,e[12]=r[0]+w-(M*w+S*x+N*T),e[13]=r[1]+x-(P*w+O*x+U*T),e[14]=r[2]+T-(F*w+R*x+L*T),e[15]=1,e}function b2(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r+r,a=n+n,c=i+i,u=r*o,l=n*o,f=n*a,h=i*o,p=i*a,d=i*c,m=s*o,g=s*a,y=s*c;return e[0]=1-f-d,e[1]=l+y,e[2]=h-g,e[3]=0,e[4]=l-y,e[5]=1-u-d,e[6]=p+m,e[7]=0,e[8]=h+g,e[9]=p-m,e[10]=1-u-f,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function I2(e,t,r,n,i,s,o){var a=1/(r-t),c=1/(i-n),u=1/(s-o);return e[0]=s*2*a,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=s*2*c,e[6]=0,e[7]=0,e[8]=(r+t)*a,e[9]=(i+n)*c,e[10]=(o+s)*u,e[11]=-1,e[12]=0,e[13]=0,e[14]=o*s*2*u,e[15]=0,e}function _2(e,t,r,n,i){var s=1/Math.tan(t/2),o;return e[0]=s/r,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=s,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=-1,e[12]=0,e[13]=0,e[15]=0,i!=null&&i!==1/0?(o=1/(n-i),e[10]=(i+n)*o,e[14]=2*i*n*o):(e[10]=-1,e[14]=-2*n),e}function C2(e,t,r,n){var i=Math.tan(t.upDegrees*Math.PI/180),s=Math.tan(t.downDegrees*Math.PI/180),o=Math.tan(t.leftDegrees*Math.PI/180),a=Math.tan(t.rightDegrees*Math.PI/180),c=2/(o+a),u=2/(i+s);return e[0]=c,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=u,e[6]=0,e[7]=0,e[8]=-((o-a)*c*.5),e[9]=(i-s)*u*.5,e[10]=n/(r-n),e[11]=-1,e[12]=0,e[13]=0,e[14]=n*r/(r-n),e[15]=0,e}function A2(e,t,r,n,i,s,o){var a=1/(t-r),c=1/(n-i),u=1/(s-o);return e[0]=-2*a,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=-2*c,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=2*u,e[11]=0,e[12]=(t+r)*a,e[13]=(i+n)*c,e[14]=(o+s)*u,e[15]=1,e}function M2(e,t,r,n){var i,s,o,a,c,u,l,f,h,p,d=t[0],m=t[1],g=t[2],y=n[0],I=n[1],_=n[2],E=r[0],b=r[1],C=r[2];return Math.abs(d-E)<Z&&Math.abs(m-b)<Z&&Math.abs(g-C)<Z?_m(e):(l=d-E,f=m-b,h=g-C,p=1/Math.hypot(l,f,h),l*=p,f*=p,h*=p,i=I*h-_*f,s=_*l-y*h,o=y*f-I*l,p=Math.hypot(i,s,o),p?(p=1/p,i*=p,s*=p,o*=p):(i=0,s=0,o=0),a=f*o-h*s,c=h*i-l*o,u=l*s-f*i,p=Math.hypot(a,c,u),p?(p=1/p,a*=p,c*=p,u*=p):(a=0,c=0,u=0),e[0]=i,e[1]=a,e[2]=l,e[3]=0,e[4]=s,e[5]=c,e[6]=f,e[7]=0,e[8]=o,e[9]=u,e[10]=h,e[11]=0,e[12]=-(i*d+s*m+o*g),e[13]=-(a*d+c*m+u*g),e[14]=-(l*d+f*m+h*g),e[15]=1,e)}function T2(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=n[0],c=n[1],u=n[2],l=i-r[0],f=s-r[1],h=o-r[2],p=l*l+f*f+h*h;p>0&&(p=1/Math.sqrt(p),l*=p,f*=p,h*=p);var d=c*h-u*f,m=u*l-a*h,g=a*f-c*l;return p=d*d+m*m+g*g,p>0&&(p=1/Math.sqrt(p),d*=p,m*=p,g*=p),e[0]=d,e[1]=m,e[2]=g,e[3]=0,e[4]=f*g-h*m,e[5]=h*d-l*g,e[6]=l*m-f*d,e[7]=0,e[8]=l,e[9]=f,e[10]=h,e[11]=0,e[12]=i,e[13]=s,e[14]=o,e[15]=1,e}function k2(e){return"mat4("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+", "+e[4]+", "+e[5]+", "+e[6]+", "+e[7]+", "+e[8]+", "+e[9]+", "+e[10]+", "+e[11]+", "+e[12]+", "+e[13]+", "+e[14]+", "+e[15]+")"}function N2(e){return Math.hypot(e[0],e[1],e[3],e[4],e[5],e[6],e[7],e[8],e[9],e[10],e[11],e[12],e[13],e[14],e[15])}function D2(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e[3]=t[3]+r[3],e[4]=t[4]+r[4],e[5]=t[5]+r[5],e[6]=t[6]+r[6],e[7]=t[7]+r[7],e[8]=t[8]+r[8],e[9]=t[9]+r[9],e[10]=t[10]+r[10],e[11]=t[11]+r[11],e[12]=t[12]+r[12],e[13]=t[13]+r[13],e[14]=t[14]+r[14],e[15]=t[15]+r[15],e}function Tm(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e[3]=t[3]-r[3],e[4]=t[4]-r[4],e[5]=t[5]-r[5],e[6]=t[6]-r[6],e[7]=t[7]-r[7],e[8]=t[8]-r[8],e[9]=t[9]-r[9],e[10]=t[10]-r[10],e[11]=t[11]-r[11],e[12]=t[12]-r[12],e[13]=t[13]-r[13],e[14]=t[14]-r[14],e[15]=t[15]-r[15],e}function P2(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e[3]=t[3]*r,e[4]=t[4]*r,e[5]=t[5]*r,e[6]=t[6]*r,e[7]=t[7]*r,e[8]=t[8]*r,e[9]=t[9]*r,e[10]=t[10]*r,e[11]=t[11]*r,e[12]=t[12]*r,e[13]=t[13]*r,e[14]=t[14]*r,e[15]=t[15]*r,e}function R2(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e[3]=t[3]+r[3]*n,e[4]=t[4]+r[4]*n,e[5]=t[5]+r[5]*n,e[6]=t[6]+r[6]*n,e[7]=t[7]+r[7]*n,e[8]=t[8]+r[8]*n,e[9]=t[9]+r[9]*n,e[10]=t[10]+r[10]*n,e[11]=t[11]+r[11]*n,e[12]=t[12]+r[12]*n,e[13]=t[13]+r[13]*n,e[14]=t[14]+r[14]*n,e[15]=t[15]+r[15]*n,e}function O2(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]&&e[3]===t[3]&&e[4]===t[4]&&e[5]===t[5]&&e[6]===t[6]&&e[7]===t[7]&&e[8]===t[8]&&e[9]===t[9]&&e[10]===t[10]&&e[11]===t[11]&&e[12]===t[12]&&e[13]===t[13]&&e[14]===t[14]&&e[15]===t[15]}function U2(e,t){var r=e[0],n=e[1],i=e[2],s=e[3],o=e[4],a=e[5],c=e[6],u=e[7],l=e[8],f=e[9],h=e[10],p=e[11],d=e[12],m=e[13],g=e[14],y=e[15],I=t[0],_=t[1],E=t[2],b=t[3],C=t[4],v=t[5],w=t[6],x=t[7],T=t[8],M=t[9],P=t[10],F=t[11],S=t[12],O=t[13],R=t[14],N=t[15];return Math.abs(r-I)<=Z*Math.max(1,Math.abs(r),Math.abs(I))&&Math.abs(n-_)<=Z*Math.max(1,Math.abs(n),Math.abs(_))&&Math.abs(i-E)<=Z*Math.max(1,Math.abs(i),Math.abs(E))&&Math.abs(s-b)<=Z*Math.max(1,Math.abs(s),Math.abs(b))&&Math.abs(o-C)<=Z*Math.max(1,Math.abs(o),Math.abs(C))&&Math.abs(a-v)<=Z*Math.max(1,Math.abs(a),Math.abs(v))&&Math.abs(c-w)<=Z*Math.max(1,Math.abs(c),Math.abs(w))&&Math.abs(u-x)<=Z*Math.max(1,Math.abs(u),Math.abs(x))&&Math.abs(l-T)<=Z*Math.max(1,Math.abs(l),Math.abs(T))&&Math.abs(f-M)<=Z*Math.max(1,Math.abs(f),Math.abs(M))&&Math.abs(h-P)<=Z*Math.max(1,Math.abs(h),Math.abs(P))&&Math.abs(p-F)<=Z*Math.max(1,Math.abs(p),Math.abs(F))&&Math.abs(d-S)<=Z*Math.max(1,Math.abs(d),Math.abs(S))&&Math.abs(m-O)<=Z*Math.max(1,Math.abs(m),Math.abs(O))&&Math.abs(g-R)<=Z*Math.max(1,Math.abs(g),Math.abs(R))&&Math.abs(y-N)<=Z*Math.max(1,Math.abs(y),Math.abs(N))}var L2=Cm,F2=Tm;var Qt={};Pi(Qt,{add:()=>sC,calculateW:()=>J_,clone:()=>tC,conjugate:()=>Z_,copy:()=>nC,create:()=>Ku,dot:()=>Hm,equals:()=>fC,exactEquals:()=>lC,exp:()=>qm,fromEuler:()=>Q_,fromMat3:()=>Ym,fromValues:()=>rC,getAngle:()=>j_,getAxisAngle:()=>G_,identity:()=>V_,invert:()=>X_,len:()=>cC,length:()=>Wm,lerp:()=>aC,ln:()=>Km,mul:()=>oC,multiply:()=>jm,normalize:()=>Yu,pow:()=>H_,random:()=>W_,rotateX:()=>q_,rotateY:()=>K_,rotateZ:()=>Y_,rotationTo:()=>hC,scale:()=>Jm,set:()=>iC,setAxes:()=>dC,setAxisAngle:()=>Gm,slerp:()=>lo,sqlerp:()=>pC,sqrLen:()=>uC,squaredLength:()=>Xm,str:()=>eC});var B={};Pi(B,{add:()=>V2,angle:()=>c_,bezier:()=>e_,ceil:()=>G2,clone:()=>B2,copy:()=>z2,create:()=>oo,cross:()=>ji,dist:()=>g_,distance:()=>Rm,div:()=>m_,divide:()=>Pm,dot:()=>ao,equals:()=>h_,exactEquals:()=>f_,floor:()=>j2,forEach:()=>x_,fromValues:()=>Mn,hermite:()=>Q2,inverse:()=>X2,len:()=>Ru,length:()=>km,lerp:()=>Z2,max:()=>K2,min:()=>q2,mul:()=>d_,multiply:()=>Dm,negate:()=>W2,normalize:()=>Gi,random:()=>t_,rotateX:()=>s_,rotateY:()=>o_,rotateZ:()=>a_,round:()=>Y2,scale:()=>J2,scaleAndAdd:()=>H2,set:()=>$2,sqrDist:()=>y_,sqrLen:()=>v_,squaredDistance:()=>Om,squaredLength:()=>Um,str:()=>l_,sub:()=>p_,subtract:()=>Nm,transformMat3:()=>n_,transformMat4:()=>r_,transformQuat:()=>i_,zero:()=>u_});function oo(){var e=new Se(3);return Se!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e}function B2(e){var t=new Se(3);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t}function km(e){var t=e[0],r=e[1],n=e[2];return Math.hypot(t,r,n)}function Mn(e,t,r){var n=new Se(3);return n[0]=e,n[1]=t,n[2]=r,n}function z2(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e}function $2(e,t,r,n){return e[0]=t,e[1]=r,e[2]=n,e}function V2(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e}function Nm(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e}function Dm(e,t,r){return e[0]=t[0]*r[0],e[1]=t[1]*r[1],e[2]=t[2]*r[2],e}function Pm(e,t,r){return e[0]=t[0]/r[0],e[1]=t[1]/r[1],e[2]=t[2]/r[2],e}function G2(e,t){return e[0]=Math.ceil(t[0]),e[1]=Math.ceil(t[1]),e[2]=Math.ceil(t[2]),e}function j2(e,t){return e[0]=Math.floor(t[0]),e[1]=Math.floor(t[1]),e[2]=Math.floor(t[2]),e}function q2(e,t,r){return e[0]=Math.min(t[0],r[0]),e[1]=Math.min(t[1],r[1]),e[2]=Math.min(t[2],r[2]),e}function K2(e,t,r){return e[0]=Math.max(t[0],r[0]),e[1]=Math.max(t[1],r[1]),e[2]=Math.max(t[2],r[2]),e}function Y2(e,t){return e[0]=Math.round(t[0]),e[1]=Math.round(t[1]),e[2]=Math.round(t[2]),e}function J2(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e}function H2(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e}function Rm(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2];return Math.hypot(r,n,i)}function Om(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2];return r*r+n*n+i*i}function Um(e){var t=e[0],r=e[1],n=e[2];return t*t+r*r+n*n}function W2(e,t){return e[0]=-t[0],e[1]=-t[1],e[2]=-t[2],e}function X2(e,t){return e[0]=1/t[0],e[1]=1/t[1],e[2]=1/t[2],e}function Gi(e,t){var r=t[0],n=t[1],i=t[2],s=r*r+n*n+i*i;return s>0&&(s=1/Math.sqrt(s)),e[0]=t[0]*s,e[1]=t[1]*s,e[2]=t[2]*s,e}function ao(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]}function ji(e,t,r){var n=t[0],i=t[1],s=t[2],o=r[0],a=r[1],c=r[2];return e[0]=i*c-s*a,e[1]=s*o-n*c,e[2]=n*a-i*o,e}function Z2(e,t,r,n){var i=t[0],s=t[1],o=t[2];return e[0]=i+n*(r[0]-i),e[1]=s+n*(r[1]-s),e[2]=o+n*(r[2]-o),e}function Q2(e,t,r,n,i,s){var o=s*s,a=o*(2*s-3)+1,c=o*(s-2)+s,u=o*(s-1),l=o*(3-2*s);return e[0]=t[0]*a+r[0]*c+n[0]*u+i[0]*l,e[1]=t[1]*a+r[1]*c+n[1]*u+i[1]*l,e[2]=t[2]*a+r[2]*c+n[2]*u+i[2]*l,e}function e_(e,t,r,n,i,s){var o=1-s,a=o*o,c=s*s,u=a*o,l=3*s*a,f=3*c*o,h=c*s;return e[0]=t[0]*u+r[0]*l+n[0]*f+i[0]*h,e[1]=t[1]*u+r[1]*l+n[1]*f+i[1]*h,e[2]=t[2]*u+r[2]*l+n[2]*f+i[2]*h,e}function t_(e,t){t=t||1;var r=Mt()*2*Math.PI,n=Mt()*2-1,i=Math.sqrt(1-n*n)*t;return e[0]=Math.cos(r)*i,e[1]=Math.sin(r)*i,e[2]=n*t,e}function r_(e,t,r){var n=t[0],i=t[1],s=t[2],o=r[3]*n+r[7]*i+r[11]*s+r[15];return o=o||1,e[0]=(r[0]*n+r[4]*i+r[8]*s+r[12])/o,e[1]=(r[1]*n+r[5]*i+r[9]*s+r[13])/o,e[2]=(r[2]*n+r[6]*i+r[10]*s+r[14])/o,e}function n_(e,t,r){var n=t[0],i=t[1],s=t[2];return e[0]=n*r[0]+i*r[3]+s*r[6],e[1]=n*r[1]+i*r[4]+s*r[7],e[2]=n*r[2]+i*r[5]+s*r[8],e}function i_(e,t,r){var n=r[0],i=r[1],s=r[2],o=r[3],a=t[0],c=t[1],u=t[2],l=i*u-s*c,f=s*a-n*u,h=n*c-i*a,p=i*h-s*f,d=s*l-n*h,m=n*f-i*l,g=o*2;return l*=g,f*=g,h*=g,p*=2,d*=2,m*=2,e[0]=a+l+p,e[1]=c+f+d,e[2]=u+h+m,e}function s_(e,t,r,n){var i=[],s=[];return i[0]=t[0]-r[0],i[1]=t[1]-r[1],i[2]=t[2]-r[2],s[0]=i[0],s[1]=i[1]*Math.cos(n)-i[2]*Math.sin(n),s[2]=i[1]*Math.sin(n)+i[2]*Math.cos(n),e[0]=s[0]+r[0],e[1]=s[1]+r[1],e[2]=s[2]+r[2],e}function o_(e,t,r,n){var i=[],s=[];return i[0]=t[0]-r[0],i[1]=t[1]-r[1],i[2]=t[2]-r[2],s[0]=i[2]*Math.sin(n)+i[0]*Math.cos(n),s[1]=i[1],s[2]=i[2]*Math.cos(n)-i[0]*Math.sin(n),e[0]=s[0]+r[0],e[1]=s[1]+r[1],e[2]=s[2]+r[2],e}function a_(e,t,r,n){var i=[],s=[];return i[0]=t[0]-r[0],i[1]=t[1]-r[1],i[2]=t[2]-r[2],s[0]=i[0]*Math.cos(n)-i[1]*Math.sin(n),s[1]=i[0]*Math.sin(n)+i[1]*Math.cos(n),s[2]=i[2],e[0]=s[0]+r[0],e[1]=s[1]+r[1],e[2]=s[2]+r[2],e}function c_(e,t){var r=Mn(e[0],e[1],e[2]),n=Mn(t[0],t[1],t[2]);Gi(r,r),Gi(n,n);var i=ao(r,n);return i>1?0:i<-1?Math.PI:Math.acos(i)}function u_(e){return e[0]=0,e[1]=0,e[2]=0,e}function l_(e){return"vec3("+e[0]+", "+e[1]+", "+e[2]+")"}function f_(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]}function h_(e,t){var r=e[0],n=e[1],i=e[2],s=t[0],o=t[1],a=t[2];return Math.abs(r-s)<=Z*Math.max(1,Math.abs(r),Math.abs(s))&&Math.abs(n-o)<=Z*Math.max(1,Math.abs(n),Math.abs(o))&&Math.abs(i-a)<=Z*Math.max(1,Math.abs(i),Math.abs(a))}var p_=Nm,d_=Dm,m_=Pm,g_=Rm,y_=Om,Ru=km,v_=Um,x_=function(){var e=oo();return function(t,r,n,i,s,o){var a,c;for(r||(r=3),n||(n=0),i?c=Math.min(i*r+n,t.length):c=t.length,a=n;a<c;a+=r)e[0]=t[a],e[1]=t[a+1],e[2]=t[a+2],s(e,e,o),t[a]=e[0],t[a+1]=e[1],t[a+2]=e[2];return t}}();var gr={};Pi(gr,{add:()=>Bu,ceil:()=>S_,clone:()=>Ou,copy:()=>Lu,create:()=>Lm,cross:()=>M_,dist:()=>L_,distance:()=>$m,div:()=>U_,divide:()=>zm,dot:()=>Vu,equals:()=>qu,exactEquals:()=>ju,floor:()=>w_,forEach:()=>$_,fromValues:()=>Uu,inverse:()=>A_,len:()=>B_,length:()=>co,lerp:()=>Gu,max:()=>b_,min:()=>E_,mul:()=>O_,multiply:()=>Bm,negate:()=>C_,normalize:()=>$u,random:()=>T_,round:()=>I_,scale:()=>zu,scaleAndAdd:()=>__,set:()=>Fu,sqrDist:()=>F_,sqrLen:()=>z_,squaredDistance:()=>Vm,squaredLength:()=>uo,str:()=>P_,sub:()=>R_,subtract:()=>Fm,transformMat4:()=>k_,transformQuat:()=>N_,zero:()=>D_});function Lm(){var e=new Se(4);return Se!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0,e[3]=0),e}function Ou(e){var t=new Se(4);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t}function Uu(e,t,r,n){var i=new Se(4);return i[0]=e,i[1]=t,i[2]=r,i[3]=n,i}function Lu(e,t){return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e}function Fu(e,t,r,n,i){return e[0]=t,e[1]=r,e[2]=n,e[3]=i,e}function Bu(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e[2]=t[2]+r[2],e[3]=t[3]+r[3],e}function Fm(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e[3]=t[3]-r[3],e}function Bm(e,t,r){return e[0]=t[0]*r[0],e[1]=t[1]*r[1],e[2]=t[2]*r[2],e[3]=t[3]*r[3],e}function zm(e,t,r){return e[0]=t[0]/r[0],e[1]=t[1]/r[1],e[2]=t[2]/r[2],e[3]=t[3]/r[3],e}function S_(e,t){return e[0]=Math.ceil(t[0]),e[1]=Math.ceil(t[1]),e[2]=Math.ceil(t[2]),e[3]=Math.ceil(t[3]),e}function w_(e,t){return e[0]=Math.floor(t[0]),e[1]=Math.floor(t[1]),e[2]=Math.floor(t[2]),e[3]=Math.floor(t[3]),e}function E_(e,t,r){return e[0]=Math.min(t[0],r[0]),e[1]=Math.min(t[1],r[1]),e[2]=Math.min(t[2],r[2]),e[3]=Math.min(t[3],r[3]),e}function b_(e,t,r){return e[0]=Math.max(t[0],r[0]),e[1]=Math.max(t[1],r[1]),e[2]=Math.max(t[2],r[2]),e[3]=Math.max(t[3],r[3]),e}function I_(e,t){return e[0]=Math.round(t[0]),e[1]=Math.round(t[1]),e[2]=Math.round(t[2]),e[3]=Math.round(t[3]),e}function zu(e,t,r){return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e[3]=t[3]*r,e}function __(e,t,r,n){return e[0]=t[0]+r[0]*n,e[1]=t[1]+r[1]*n,e[2]=t[2]+r[2]*n,e[3]=t[3]+r[3]*n,e}function $m(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2],s=t[3]-e[3];return Math.hypot(r,n,i,s)}function Vm(e,t){var r=t[0]-e[0],n=t[1]-e[1],i=t[2]-e[2],s=t[3]-e[3];return r*r+n*n+i*i+s*s}function co(e){var t=e[0],r=e[1],n=e[2],i=e[3];return Math.hypot(t,r,n,i)}function uo(e){var t=e[0],r=e[1],n=e[2],i=e[3];return t*t+r*r+n*n+i*i}function C_(e,t){return e[0]=-t[0],e[1]=-t[1],e[2]=-t[2],e[3]=-t[3],e}function A_(e,t){return e[0]=1/t[0],e[1]=1/t[1],e[2]=1/t[2],e[3]=1/t[3],e}function $u(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r*r+n*n+i*i+s*s;return o>0&&(o=1/Math.sqrt(o)),e[0]=r*o,e[1]=n*o,e[2]=i*o,e[3]=s*o,e}function Vu(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]+e[3]*t[3]}function M_(e,t,r,n){var i=r[0]*n[1]-r[1]*n[0],s=r[0]*n[2]-r[2]*n[0],o=r[0]*n[3]-r[3]*n[0],a=r[1]*n[2]-r[2]*n[1],c=r[1]*n[3]-r[3]*n[1],u=r[2]*n[3]-r[3]*n[2],l=t[0],f=t[1],h=t[2],p=t[3];return e[0]=f*u-h*c+p*a,e[1]=-(l*u)+h*o-p*s,e[2]=l*c-f*o+p*i,e[3]=-(l*a)+f*s-h*i,e}function Gu(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=t[3];return e[0]=i+n*(r[0]-i),e[1]=s+n*(r[1]-s),e[2]=o+n*(r[2]-o),e[3]=a+n*(r[3]-a),e}function T_(e,t){t=t||1;var r,n,i,s,o,a;do r=Mt()*2-1,n=Mt()*2-1,o=r*r+n*n;while(o>=1);do i=Mt()*2-1,s=Mt()*2-1,a=i*i+s*s;while(a>=1);var c=Math.sqrt((1-o)/a);return e[0]=t*r,e[1]=t*n,e[2]=t*i*c,e[3]=t*s*c,e}function k_(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3];return e[0]=r[0]*n+r[4]*i+r[8]*s+r[12]*o,e[1]=r[1]*n+r[5]*i+r[9]*s+r[13]*o,e[2]=r[2]*n+r[6]*i+r[10]*s+r[14]*o,e[3]=r[3]*n+r[7]*i+r[11]*s+r[15]*o,e}function N_(e,t,r){var n=t[0],i=t[1],s=t[2],o=r[0],a=r[1],c=r[2],u=r[3],l=u*n+a*s-c*i,f=u*i+c*n-o*s,h=u*s+o*i-a*n,p=-o*n-a*i-c*s;return e[0]=l*u+p*-o+f*-c-h*-a,e[1]=f*u+p*-a+h*-o-l*-c,e[2]=h*u+p*-c+l*-a-f*-o,e[3]=t[3],e}function D_(e){return e[0]=0,e[1]=0,e[2]=0,e[3]=0,e}function P_(e){return"vec4("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+")"}function ju(e,t){return e[0]===t[0]&&e[1]===t[1]&&e[2]===t[2]&&e[3]===t[3]}function qu(e,t){var r=e[0],n=e[1],i=e[2],s=e[3],o=t[0],a=t[1],c=t[2],u=t[3];return Math.abs(r-o)<=Z*Math.max(1,Math.abs(r),Math.abs(o))&&Math.abs(n-a)<=Z*Math.max(1,Math.abs(n),Math.abs(a))&&Math.abs(i-c)<=Z*Math.max(1,Math.abs(i),Math.abs(c))&&Math.abs(s-u)<=Z*Math.max(1,Math.abs(s),Math.abs(u))}var R_=Fm,O_=Bm,U_=zm,L_=$m,F_=Vm,B_=co,z_=uo,$_=function(){var e=Lm();return function(t,r,n,i,s,o){var a,c;for(r||(r=4),n||(n=0),i?c=Math.min(i*r+n,t.length):c=t.length,a=n;a<c;a+=r)e[0]=t[a],e[1]=t[a+1],e[2]=t[a+2],e[3]=t[a+3],s(e,e,o),t[a]=e[0],t[a+1]=e[1],t[a+2]=e[2],t[a+3]=e[3];return t}}();function Ku(){var e=new Se(4);return Se!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e[3]=1,e}function V_(e){return e[0]=0,e[1]=0,e[2]=0,e[3]=1,e}function Gm(e,t,r){r=r*.5;var n=Math.sin(r);return e[0]=n*t[0],e[1]=n*t[1],e[2]=n*t[2],e[3]=Math.cos(r),e}function G_(e,t){var r=Math.acos(t[3])*2,n=Math.sin(r/2);return n>Z?(e[0]=t[0]/n,e[1]=t[1]/n,e[2]=t[2]/n):(e[0]=1,e[1]=0,e[2]=0),r}function j_(e,t){var r=Hm(e,t);return Math.acos(2*r*r-1)}function jm(e,t,r){var n=t[0],i=t[1],s=t[2],o=t[3],a=r[0],c=r[1],u=r[2],l=r[3];return e[0]=n*l+o*a+i*u-s*c,e[1]=i*l+o*c+s*a-n*u,e[2]=s*l+o*u+n*c-i*a,e[3]=o*l-n*a-i*c-s*u,e}function q_(e,t,r){r*=.5;var n=t[0],i=t[1],s=t[2],o=t[3],a=Math.sin(r),c=Math.cos(r);return e[0]=n*c+o*a,e[1]=i*c+s*a,e[2]=s*c-i*a,e[3]=o*c-n*a,e}function K_(e,t,r){r*=.5;var n=t[0],i=t[1],s=t[2],o=t[3],a=Math.sin(r),c=Math.cos(r);return e[0]=n*c-s*a,e[1]=i*c+o*a,e[2]=s*c+n*a,e[3]=o*c-i*a,e}function Y_(e,t,r){r*=.5;var n=t[0],i=t[1],s=t[2],o=t[3],a=Math.sin(r),c=Math.cos(r);return e[0]=n*c+i*a,e[1]=i*c-n*a,e[2]=s*c+o*a,e[3]=o*c-s*a,e}function J_(e,t){var r=t[0],n=t[1],i=t[2];return e[0]=r,e[1]=n,e[2]=i,e[3]=Math.sqrt(Math.abs(1-r*r-n*n-i*i)),e}function qm(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=Math.sqrt(r*r+n*n+i*i),a=Math.exp(s),c=o>0?a*Math.sin(o)/o:0;return e[0]=r*c,e[1]=n*c,e[2]=i*c,e[3]=a*Math.cos(o),e}function Km(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=Math.sqrt(r*r+n*n+i*i),a=o>0?Math.atan2(o,s)/o:0;return e[0]=r*a,e[1]=n*a,e[2]=i*a,e[3]=.5*Math.log(r*r+n*n+i*i+s*s),e}function H_(e,t,r){return Km(e,t),Jm(e,e,r),qm(e,e),e}function lo(e,t,r,n){var i=t[0],s=t[1],o=t[2],a=t[3],c=r[0],u=r[1],l=r[2],f=r[3],h,p,d,m,g;return p=i*c+s*u+o*l+a*f,p<0&&(p=-p,c=-c,u=-u,l=-l,f=-f),1-p>Z?(h=Math.acos(p),d=Math.sin(h),m=Math.sin((1-n)*h)/d,g=Math.sin(n*h)/d):(m=1-n,g=n),e[0]=m*i+g*c,e[1]=m*s+g*u,e[2]=m*o+g*l,e[3]=m*a+g*f,e}function W_(e){var t=Mt(),r=Mt(),n=Mt(),i=Math.sqrt(1-t),s=Math.sqrt(t);return e[0]=i*Math.sin(2*Math.PI*r),e[1]=i*Math.cos(2*Math.PI*r),e[2]=s*Math.sin(2*Math.PI*n),e[3]=s*Math.cos(2*Math.PI*n),e}function X_(e,t){var r=t[0],n=t[1],i=t[2],s=t[3],o=r*r+n*n+i*i+s*s,a=o?1/o:0;return e[0]=-r*a,e[1]=-n*a,e[2]=-i*a,e[3]=s*a,e}function Z_(e,t){return e[0]=-t[0],e[1]=-t[1],e[2]=-t[2],e[3]=t[3],e}function Ym(e,t){var r=t[0]+t[4]+t[8],n;if(r>0)n=Math.sqrt(r+1),e[3]=.5*n,n=.5/n,e[0]=(t[5]-t[7])*n,e[1]=(t[6]-t[2])*n,e[2]=(t[1]-t[3])*n;else{var i=0;t[4]>t[0]&&(i=1),t[8]>t[i*3+i]&&(i=2);var s=(i+1)%3,o=(i+2)%3;n=Math.sqrt(t[i*3+i]-t[s*3+s]-t[o*3+o]+1),e[i]=.5*n,n=.5/n,e[3]=(t[s*3+o]-t[o*3+s])*n,e[s]=(t[s*3+i]+t[i*3+s])*n,e[o]=(t[o*3+i]+t[i*3+o])*n}return e}function Q_(e,t,r,n){var i=.5*Math.PI/180;t*=i,r*=i,n*=i;var s=Math.sin(t),o=Math.cos(t),a=Math.sin(r),c=Math.cos(r),u=Math.sin(n),l=Math.cos(n);return e[0]=s*c*l-o*a*u,e[1]=o*a*l+s*c*u,e[2]=o*c*u-s*a*l,e[3]=o*c*l+s*a*u,e}function eC(e){return"quat("+e[0]+", "+e[1]+", "+e[2]+", "+e[3]+")"}var tC=Ou,rC=Uu,nC=Lu,iC=Fu,sC=Bu,oC=jm,Jm=zu,Hm=Vu,aC=Gu,Wm=co,cC=Wm,Xm=uo,uC=Xm,Yu=$u,lC=ju,fC=qu,hC=function(){var e=oo(),t=Mn(1,0,0),r=Mn(0,1,0);return function(n,i,s){var o=ao(i,s);return o<-.999999?(ji(e,t,i),Ru(e)<1e-6&&ji(e,r,i),Gi(e,e),Gm(n,e,Math.PI),n):o>.999999?(n[0]=0,n[1]=0,n[2]=0,n[3]=1,n):(ji(e,i,s),n[0]=e[0],n[1]=e[1],n[2]=e[2],n[3]=1+o,Yu(n,n))}}(),pC=function(){var e=Ku(),t=Ku();return function(r,n,i,s,o,a){return lo(e,n,o,a),lo(t,i,s,a),lo(r,e,t,2*a*(1-a)),r}}(),dC=function(){var e=Pu();return function(t,r,n,i){return e[0]=n[0],e[3]=n[1],e[6]=n[2],e[1]=i[0],e[4]=i[1],e[7]=i[2],e[2]=-r[0],e[5]=-r[1],e[8]=-r[2],Yu(t,Ym(t,e))}}();function Zm(e,t){let r=e.length,n=0;for(let i=0;i<r;++i)t(e[i],i,e)&&(e[n]=e[i],++n);e.length=n}function Qm(e,t,r){let n=new e.constructor(e.length);for(let i=0;i<t*r;i+=r)for(let s=0;s<r;s++){let o=i/r;n[s*t+o]=e[i+s]}return n}function er(e,t,r,n=0,i=e.length){for(;n<i;){let s=n+i-1>>1,o=r(t,e[s]);if(o>0)n=s+1;else if(o<0)i=s;else return s}return~n}function we(e,t,r){let n=t-e;for(;n>0;){let i=Math.floor(n/2),s=e+i;r(s)?n=i:(e=s+1,n-=i+1)}return e}function qt(e,t){let r=e.length;if(t.length!==r)return!1;for(let n=0;n<r;++n)if(e[n]!==t[n])return!1;return!0}var TR=ge.create();var kR=[B.fromValues(1,0,0),B.fromValues(0,1,0),B.fromValues(0,0,1)],qi=B.fromValues(0,0,0),NR=gr.fromValues(0,0,0,0),eg=B.fromValues(1,1,1),tg=B.fromValues(1/0,1/0,1/0),DR=Qt.create();function Ki(e){return e[0]*e[1]*e[2]}function tr(e){return`${e[0]},${e[1]},${e[2]}`}function rg(e,t,r){let n=t[0],i=t[1],s=t[2];return e[0]=r[0]*n+r[4]*i+r[8]*s,e[1]=r[1]*n+r[5]*i+r[9]*s,e[2]=r[2]*n+r[6]*i+r[10]*s,e}function ng(e,t,r){let n=t[0],i=t[1],s=t[2];return e[0]=r[0]*n+r[1]*i+r[2]*s,e[1]=r[4]*n+r[5]*i+r[6]*s,e[2]=r[8]*n+r[9]*i+r[10]*s,e}function ig(e,t,r,n,i){let s=e;return e[0]=n[0],e[1]=n[1],e[2]=n[2]*i,ge.fromRotationTranslationScale(e,r,t,s)}function fo(e,t){let r=t[0],n=t[1],i=t[2],s=t[4],o=t[5],a=t[6],c=t[8],u=t[9],l=t[10];return e[0]=r,e[1]=n,e[2]=i,e[3]=s,e[4]=o,e[5]=a,e[6]=c,e[7]=u,e[8]=l,e}function ho(e,t){let r=t[0],n=t[1],i=t[2],s=t[3],o=t[4],a=t[5],c=t[6],u=t[7],l=t[8],f=t[9],h=t[10],p=t[11],d=t[12],m=t[13],g=t[14],y=t[15];e[0]=s+r,e[1]=u+o,e[2]=p+l,e[3]=y+d,e[4]=s-r,e[5]=u-o,e[6]=p-l,e[7]=y-d,e[8]=s+n,e[9]=u+a,e[10]=p+f,e[11]=y+m,e[12]=s-n,e[13]=u-a,e[14]=p-f,e[15]=y-m;let I=s+i,_=u+c,E=p+h,b=y+g,C=s-i,v=u-c,w=p-h,x=y-g,T=Math.sqrt(I**2+_**2+E**2);e[16]=I/T,e[17]=_/T,e[18]=E/T,e[19]=b/T;let M=Math.sqrt(C**2+v**2+w**2);return e[20]=C/M,e[21]=v/M,e[22]=w/M,e[23]=x/M,e}function po(e,t,r,n,i,s,o){for(let a=0;a<6;++a){let c=o[a*4],u=o[a*4+1],l=o[a*4+2],f=o[a*4+3];if(Math.max(c*e,c*n)+Math.max(u*t,u*i)+Math.max(l*r,l*s)+f<0)return!1}return!0}function sg(e,t,r,n,i,s,o){for(let a=0;a<4;++a){let c=o[a*4],u=o[a*4+1],l=o[a*4+2],f=o[a*4+3];if(Math.max(c*e,c*n)+Math.max(u*t,u*i)+Math.max(l*r,l*s)+f<0)return!1}{let c=o[20],u=o[5*4+1],l=o[5*4+2],f=o[5*4+3],h=Math.max(c*e,c*n)+Math.max(u*t,u*i)+Math.max(l*r,l*s),p=Math.min(c*e,c*n)+Math.min(u*t,u*i)+Math.min(l*r,l*s),d=Math.abs(f)*1e-6;if(p>-f+d||h<-f-d)return!1}return!0}function og(e){if(e[15]===1){let o=2/Math.abs(e[10]),a=2/Math.abs(e[0]),c=2/Math.abs(e[5]);return a*c*o}let t=e[10],n=2*e[14]/(2*t-2),i=(t-1)*n/(t+1);return 4/(e[0]*e[5])/3*(Math.abs(i)**3-Math.abs(n)**3)}function mo(e){if(e[15]===1)return 2/Math.abs(e[10]);let t=e[10],n=2*e[14]/(2*t-2),i=(t-1)*n/(t+1);return Math.abs(i-n)}var PR=B.create();function go(e){let t=typeof e;if(t==="number"||t==="string"){let r=parseFloat(""+e);if(!Number.isNaN(r))return r}throw new Error(`Expected floating-point number, but received: ${JSON.stringify(e)}.`)}function rr(e){let t=go(e);if(Number.isFinite(t))return t;throw new Error(`Expected finite floating-point number, but received: ${t}.`)}function ag(e){let t=go(e);if(Number.isFinite(t)&&t>=0)return t;throw new Error(`Expected finite non-negative floating-point number, but received: ${t}.`)}function Kt(e){if(typeof e=="object"){if(e===null)return"null";if(Array.isArray(e)){let s="[",o=e.length,a=0;if(a<o)for(s+=Kt(e[a]);++a<o;)s+=",",s+=Kt(e[a]);return s+="]",s}let t="{",r=Object.keys(e).sort(),n=0,i=r.length;if(n<i){let s=r[n];for(t+=JSON.stringify(s),t+=":",t+=Kt(e[s]);++n<i;)t+=",",s=r[n],t+=JSON.stringify(s),t+=":",t+=Kt(e[s])}return t+="}",t}return typeof e=="bigint"?e.toString():JSON.stringify(e)}var cg=/('(?:[^'\\]|(?:\\.))*')/,ug=/("(?:[^"\\]|(?:\\.))*")/,mC=new RegExp(`${cg.source}|${ug.source}`),FR=new RegExp(`${ug.source}|${cg.source}`),gC=/^((?:[^"'\\]|(?:\\[^']))*)("|\\')/;function yC(e,t,r,n){if(e.length>=2&&e.charAt(0)===t&&e.charAt(e.length-1)===t){let i=e.substr(1,e.length-2),s=r;for(;i.length>0;){let o=i.match(n);if(o===null){s+=i;break}s+=o[1],o[2]===r?(s+="\\",s+=r):s+=t,i=i.substr(o.index+o[0].length)}return s+=r,s}return e}function vC(e){return yC(e,"'",'"',gC)}function xC(e){let t="";for(;e.length>0;){let r=e.match(mC),n,i;if(r===null)n=e,e="",i="";else{n=e.substr(0,r.index),e=e.substr(r.index+r[0].length);let s=r[1];s!==void 0?i=vC(s):i=r[2]}t+=n.replace(/\(/g,"[").replace(/\)/g,"]").replace("True","true").replace("False","false").replace(/,\s*([}\]])/g,"$1"),t+=i}return t}function lg(e){return JSON.parse(xC(e))}function ct(e,t){if(!Array.isArray(e))throw new Error(`Expected array, but received: ${JSON.stringify(e)}.`);return e.map(t)}function We(e,t,r){let n=e.length;if(!Array.isArray(t)||t.length!==n)throw new Error(`Expected length ${n} array, but received: ${JSON.stringify(t)}.`);for(let i=0;i<n;++i)e[i]=r(t[i],i);return e}function Ee(e){if(typeof e!="object"||e==null||Array.isArray(e))throw new Error(`Expected JSON object, but received: ${JSON.stringify(e)}.`);return e}function kr(e){let t=parseInt(e,10);if(!Number.isInteger(t))throw new Error(`Expected integer, but received: ${JSON.stringify(e)}.`);return t}function et(e){if(typeof e!="string")throw new Error(`Expected string, but received: ${JSON.stringify(e)}.`);return e}function Ju(e){if(e!==void 0)return et(e)}function re(e,t,r){let n=Object.prototype.hasOwnProperty.call(e,t)?e[t]:void 0;try{return r(n)}catch(i){throw new Error(`Error parsing ${JSON.stringify(t)} property: ${i.message}`)}}function nr(e,t,r,n){return re(e,t,i=>i===void 0?n:r(i))}function Tn(e,t,r=/^[a-zA-Z]/){if(typeof e=="string"&&e.match(r)!==null){let n=e.toUpperCase();if(Object.prototype.hasOwnProperty.call(t,n))return t[n]}throw new Error(`Invalid enum value: ${JSON.stringify(e)}.`)}function Yt(e){if(!Array.isArray(e))throw new Error(`Expected array, received: ${JSON.stringify(e)}.`);for(let t of e)if(typeof t!="string")throw new Error(`Expected string, received: ${JSON.stringify(t)}.`);return e}function yr(e){let t;switch(typeof e){case"string":if(e.match(/^(?:0|[1-9][0-9]*)$/)===null)throw new Error(`Expected base-10 number, but received: ${JSON.stringify(e)}`);t=BigInt(e);break;case"number":t=BigInt(e);break;case"bigint":t=e;break;default:throw new Error(`Expected uint64 value, but received: ${JSON.stringify(e)}`)}if(t<0n||t>Du)throw new Error(`Expected uint64 value, but received: ${t}`);return t}var yo=class{map=new Map;get(t,r){let{map:n}=this,i=n.get(t);return i===void 0?(i=r(),i.registerDisposer(()=>{n.delete(t)}),n.set(t,i)):i.addRef(),i}},kn=class extends yo{get(t,r){return typeof t!="string"&&(t=Kt(t)),super.get(t,r)}getUncounted(t,r){return this.get(t,()=>new Xs(r())).value}getAsync(t,r,n){return this.getUncounted(t,()=>Yi(n))(r)}};function Yi(e){let t,r,n,i=!1;return async s=>{if(i)return n;let{signal:o}=s;if(o?.throwIfAborted(),n===void 0||r.signal.aborted){t=new eo,r=new Zs;let c=r;n=(async()=>{try{return await e({signal:c.signal,progressListener:t})}catch(u){throw c.signal.aborted&&(n=void 0),u}finally{n!==void 0&&(i=!0),t=void 0,c[Symbol.dispose](),r===c&&(r=void 0)}})()}r.addConsumer(o);let a=t;a.addListener(s.progressListener);try{return await Qs(n,o)}finally{a.removeListener(s.progressListener)}}}function vo(e){let{child:t,next:r,prev:n,compare:i}=e;function s(f){let h=f[t];if(h===null)return null;let p=null;for(;;){let m=h[r],g,y;if(m===null?(g=null,y=h):(g=m[r],y=o(h,m)),y[r]=p,p=y,g===null)break;h=g}let d=p;for(p=p[r];p!==null;){let m=p[r];d=o(d,p),p=m}return d[n]=null,d[r]=null,d}function o(f,h){if(h===null)return f;if(f===null)return h;if(i(h,f)){let d=f;f=h,h=d}let p=f[t];return h[r]=p,h[n]=f,p!==null&&(p[n]=h),f[t]=h,f}function a(f){let h=s(f);return f[r]=null,f[n]=null,f[t]=null,h}function c(f,h){if(f===h)return a(f);let p=h[n],d=h[r];p[t]===h?p[t]=d:p[r]=d,d!==null&&(d[n]=p);let m=o(f,s(h));return h[r]=null,h[n]=null,h[t]=null,m}function*u(f){if(f!==null){let h=f[t];for(yield f;h!==null;){let p=h[r];yield*u(h),h=p}}}function*l(f){if(f!==null){let h=f[t];for(f[t]=null,f[r]=null,f[n]=null,yield f;h!==null;){let p=h[r];h[t]=null,h[r]=null,h[n]=null,yield*u(h),h=p}}}return{compare:i,meld:o,removeMin:a,remove:c,entries:u,removedEntries:l}}var SC=Object.defineProperty,wC=Object.getOwnPropertyDescriptor,Xu=(e,t,r,n)=>{for(var i=n>1?void 0:n?wC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&SC(t,r,i),i};var Ji=!1,EC=0;function mg(){return++EC}var fe=class{child0=null;next0=null;prev0=null;child1=null;next1=null;prev1=null;source=null;key=null;state_=z.NEW;error=null;markGeneration=-1;priority=0;newPriority=0;priorityTier=se.RECENT;newPriorityTier=se.RECENT;systemMemoryBytes_=0;gpuMemoryBytes_=0;downloadSlots_=1;isComputational=!1;requestedState=z.NEW;newRequestedState=z.NEW;downloadAbortController=void 0;initialize(t){this.key=t,this.priority=Number.NEGATIVE_INFINITY,this.priorityTier=se.RECENT,this.newPriority=Number.NEGATIVE_INFINITY,this.newPriorityTier=se.RECENT,this.error=null,this.state=z.NEW,this.requestedState=z.NEW,this.newRequestedState=z.NEW}updatePriorityProperties(){this.priorityTier=this.newPriorityTier,this.priority=this.newPriority,this.newPriorityTier=se.RECENT,this.newPriority=Number.NEGATIVE_INFINITY,this.requestedState=this.newRequestedState,this.newRequestedState=z.NEW}dispose(){this.source=null,this.error=null}get chunkManager(){return this.source.chunkManager}get queueManager(){return this.source.chunkManager.queueManager}downloadFailed(t){this.error=t,this.queueManager.updateChunkState(this,z.FAILED)}downloadSucceeded(){this.requestedState===z.SYSTEM_MEMORY?(this.queueManager.moveChunkToFrontend(this),this.queueManager.updateChunkState(this,z.SYSTEM_MEMORY)):this.queueManager.updateChunkState(this,z.SYSTEM_MEMORY_WORKER)}freeSystemMemory(){}serialize(t,r){t.id=this.key,t.source=this.source.rpcId,t.new=!0}toString(){return this.key}set state(t){if(t===this.state_)return;let r=this.state_;this.state_=t,this.source.chunkStateChanged(this,r)}get state(){return this.state_}set systemMemoryBytes(t){vr(this,-1),this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!1),this.systemMemoryBytes_=t,this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!0),vr(this,1),this.chunkManager.queueManager.scheduleUpdate()}get systemMemoryBytes(){return this.systemMemoryBytes_}set gpuMemoryBytes(t){vr(this,-1),this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!1),this.gpuMemoryBytes_=t,this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!0),vr(this,1),this.chunkManager.queueManager.scheduleUpdate()}get gpuMemoryBytes(){return this.gpuMemoryBytes_}get downloadSlots(){return this.downloadSlots_}set downloadSlots(t){t!==this.downloadSlots_&&(vr(this,-1),this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!1),this.downloadSlots_=t,this.chunkManager.queueManager.adjustCapacitiesForChunk(this,!0),vr(this,1),this.chunkManager.queueManager.scheduleUpdate())}registerListener(t){return this.source?this.source.registerChunkListener(this.key,t):!1}unregisterListener(t){return this.source?this.source.unregisterChunkListener(this.key,t):!1}static priorityLess(t,r){return t.priority<r.priority}static priorityGreater(t,r){return t.priority>r.priority}},bC=2,Hi=class extends rn{constructor(t){super(),this.chunkManager=t,t.queueManager.sources.add(this)}listeners_=new Map;chunks=new Map;freeChunks=new Array;statistics=new Float64Array(mm);sourceQueueLevel=0;disposed(){this.chunkManager.queueManager.sources.delete(this),super.disposed()}getNewChunk_(t){let r=this.freeChunks,n=r.length;if(n>0){let s=r[n-1];return r.length=n-1,s.source=this,s}let i=new t;return i.source=this,i}addChunk(t){let{chunks:r}=this;r.size===0&&this.addRef(),r.set(t.key,t),vr(t,1)}removeChunk(t){let{chunks:r,freeChunks:n}=this;r.delete(t.key),t.dispose(),n[n.length]=t,r.size===0&&this.dispose()}registerChunkListener(t,r){return this.listeners_.has(t)?this.listeners_.get(t).push(r):this.listeners_.set(t,[r]),!0}unregisterChunkListener(t,r){if(!this.listeners_.has(t))return!1;let n=this.listeners_.get(t),i=n.indexOf(r);return i<0?!1:(n.splice(i,1),n.length===0&&this.listeners_.delete(t),!0)}chunkStateChanged(t,r){let{key:n}=t;if(n===null)return;let i=this.listeners_.get(n);if(i!==void 0)for(let s of i.slice())s(t,r)}};function vr(e,t){let{statistics:r}=e.source,{systemMemoryBytes:n,gpuMemoryBytes:i}=e,s=gm(e.state,e.priorityTier);r[s*An+$i.numChunks]+=t,r[s*An+$i.systemMemoryBytes]+=t*n,r[s*An+$i.gpuMemoryBytes]+=t*i}var Ne=class extends Hi{constructor(t,r){let n=t.get(r.chunkManager);super(n),Mu(this,t,r)}};function IC(e){let t=e.downloadAbortController=new AbortController,r=Date.now();e.source.download(e,t.signal).then(()=>{if(e.downloadAbortController===t){e.downloadAbortController=void 0;let n=Date.now(),{statistics:i}=e.source;i[ku(io.totalTime)]+=n-r,++i[ku(io.totalChunks)],e.downloadSucceeded()}},n=>{e.downloadAbortController===t&&(e.downloadAbortController=void 0,e.downloadFailed(n),console.log(`Error retrieving chunk ${e}: ${n}`))})}function fg(e){let t=e.downloadAbortController;e.downloadAbortController=void 0,t.abort(new DOMException("chunk download cancelled","AbortError"))}var xo=class{constructor(t,r){this.heapOperations=t,this.linkedListOperations=r,r.initializeHead(this.recentHead)}heapRoots=[null,null];recentHead=new fe;add(t){let r=t.priorityTier;if(r===se.RECENT)this.linkedListOperations.insertAfter(this.recentHead,t);else{let{heapRoots:n}=this;n[r]=this.heapOperations.meld(n[r],t)}}*candidates(){if(this.heapOperations.compare===fe.priorityLess){let{linkedListOperations:t,recentHead:r}=this;for(;;){let i=t.back(r);if(i==null)break;yield i}let{heapRoots:n}=this;for(let i=se.LAST_ORDERED_TIER;i>=se.FIRST_ORDERED_TIER;--i)for(;;){let s=n[i];if(s==null)break;yield s}}else{let t=this.heapRoots;for(let i=se.FIRST_ORDERED_TIER;i<=se.LAST_ORDERED_TIER;++i)for(;;){let s=t[i];if(s==null)break;yield s}let{linkedListOperations:r,recentHead:n}=this;for(;;){let i=r.front(n);if(i==null)break;yield i}}}delete(t){let r=t.priorityTier;if(r===se.RECENT)this.linkedListOperations.pop(t);else{let n=this.heapRoots;n[r]=this.heapOperations.remove(n[r],t)}}},_C=so({next:"next0",prev:"prev0"}),CC=so({next:"next1",prev:"prev1"});function AC(e){return new xo(vo({compare:e,child:"child0",next:"next0",prev:"prev0"}),_C)}function Nr(e){return new xo(vo({compare:e,child:"child1",next:"next1",prev:"prev1"}),CC)}function Hu(e,t,r,n,i,s){for(;t.availableItems<1||t.availableSize<e;){let o=i.next().value;if(o===void 0)return!1;let a=o.priorityTier;if(a<r||a===r&&o.priority>=n)return!1;s(o)}return!0}var Wu=class extends be{constructor(t,r){super(),this.itemLimit=t,this.sizeLimit=r,this.registerDisposer(t.changed.add(this.capacityChanged.dispatch)),this.registerDisposer(r.changed.add(this.capacityChanged.dispatch))}currentSize=0;currentItems=0;capacityChanged=new $e;adjust(t,r){this.currentItems-=t,this.currentSize-=r}get availableSize(){return this.sizeLimit.value-this.currentSize}get availableItems(){return this.itemLimit.value-this.currentItems}toString(){return`bytes=${this.currentSize}/${this.sizeLimit.value},items=${this.currentItems}/${this.itemLimit.value}`}},hg=class extends me{gpuMemoryCapacity;systemMemoryCapacity;downloadCapacity;computeCapacity;enablePrefetch;sources=new Set;queuedDownloadPromotionQueue=[Nr(fe.priorityGreater),Nr(fe.priorityGreater)];queuedComputePromotionQueue=Nr(fe.priorityGreater);downloadEvictionQueue=[Nr(fe.priorityLess),Nr(fe.priorityLess)];computeEvictionQueue=Nr(fe.priorityLess);systemMemoryEvictionQueue=AC(fe.priorityLess);gpuMemoryPromotionQueue=Nr(fe.priorityGreater);gpuMemoryEvictionQueue=Nr(fe.priorityLess);updatePending=null;gpuMemoryChanged=new $e;numQueued=0;numFailed=0;gpuMemoryGeneration=0;constructor(e,t){super(e,t);let r=n=>{let i=this.registerDisposer(new Wu(e.get(n.itemLimit),e.get(n.sizeLimit)));return i.capacityChanged.add(()=>this.scheduleUpdate()),i};this.gpuMemoryCapacity=r(t.gpuMemoryCapacity),this.systemMemoryCapacity=r(t.systemMemoryCapacity),this.enablePrefetch=e.get(t.enablePrefetch),this.downloadCapacity=[r(t.downloadCapacity),r(t.downloadCapacity)],this.computeCapacity=r(t.computeCapacity)}scheduleUpdate(){this.updatePending===null&&(this.updatePending=setTimeout(this.process.bind(this),0))}*chunkQueuesForChunk(e){switch(e.state){case z.QUEUED:e.isComputational?yield this.queuedComputePromotionQueue:yield this.queuedDownloadPromotionQueue[e.source.sourceQueueLevel];break;case z.DOWNLOADING:e.isComputational?yield this.computeEvictionQueue:(yield this.downloadEvictionQueue[e.source.sourceQueueLevel],yield this.systemMemoryEvictionQueue);break;case z.SYSTEM_MEMORY_WORKER:case z.SYSTEM_MEMORY:yield this.systemMemoryEvictionQueue,e.requestedState===z.GPU_MEMORY&&(yield this.gpuMemoryPromotionQueue);break;case z.GPU_MEMORY:yield this.systemMemoryEvictionQueue,yield this.gpuMemoryEvictionQueue;break}}adjustCapacitiesForChunk(e,t){let r=t?-1:1;switch(e.state){case z.FAILED:this.numFailed-=r;break;case z.QUEUED:this.numQueued-=r;break;case z.DOWNLOADING:(e.isComputational?this.computeCapacity:this.downloadCapacity[e.source.sourceQueueLevel]).adjust(r*e.downloadSlots,r*e.systemMemoryBytes),this.systemMemoryCapacity.adjust(r,r*e.systemMemoryBytes);break;case z.SYSTEM_MEMORY:case z.SYSTEM_MEMORY_WORKER:this.systemMemoryCapacity.adjust(r,r*e.systemMemoryBytes);break;case z.GPU_MEMORY:this.systemMemoryCapacity.adjust(r,r*e.systemMemoryBytes),this.gpuMemoryCapacity.adjust(r,r*e.gpuMemoryBytes);break}}removeChunkFromQueues_(e){vr(e,-1);for(let t of this.chunkQueuesForChunk(e))t.delete(e)}addChunkToQueues_(e){if(e.state===z.QUEUED&&e.priorityTier===se.RECENT){let{source:t}=e;return t.removeChunk(e),this.adjustCapacitiesForChunk(e,!1),!1}vr(e,1);for(let t of this.chunkQueuesForChunk(e))t.add(e);return!0}performChunkPriorityUpdate(e){if(e.priorityTier===e.newPriorityTier&&e.priority===e.newPriority){e.newPriorityTier=se.RECENT,e.newPriority=Number.NEGATIVE_INFINITY;return}Ji&&console.log(`${e}: changed priority ${e.priorityTier}:${e.priority} -> ${e.newPriorityTier}:${e.newPriority}`),this.removeChunkFromQueues_(e),e.updatePriorityProperties(),e.state===z.NEW&&(e.state=z.QUEUED,this.adjustCapacitiesForChunk(e,!0)),this.addChunkToQueues_(e)}updateChunkState(e,t){t!==e.state&&(Ji&&console.log(`${e}: changed state ${z[e.state]} -> ${z[t]}`),this.adjustCapacitiesForChunk(e,!1),this.removeChunkFromQueues_(e),e.state=t,this.adjustCapacitiesForChunk(e,!0),this.addChunkToQueues_(e),this.scheduleUpdate())}markRecentlyUsed(e){this.removeChunkFromQueues_(e),this.addChunkToQueues_(e)}processGPUPromotions_(){let e=this;function t(s){e.freeChunkGPUMemory(s),s.source.chunkManager.queueManager.updateChunkState(s,z.SYSTEM_MEMORY)}let r=this.gpuMemoryPromotionQueue.candidates(),n=this.gpuMemoryEvictionQueue.candidates(),i=this.gpuMemoryCapacity;for(;;){let s=r.next().value;if(s===void 0)break;let o=s.priorityTier,a=s.priority;if(!Hu(s.gpuMemoryBytes,i,o,a,n,t))break;this.copyChunkToGPU(s),this.updateChunkState(s,z.GPU_MEMORY)}}freeChunkGPUMemory(e){++this.gpuMemoryGeneration,this.rpc.invoke("Chunk.update",{id:e.key,state:z.SYSTEM_MEMORY,source:e.source.rpcId})}freeChunkSystemMemory(e){e.state===z.SYSTEM_MEMORY_WORKER?e.freeSystemMemory():this.rpc.invoke("Chunk.update",{id:e.key,state:z.EXPIRED,source:e.source.rpcId})}retrieveChunkData(e){return this.rpc.promiseInvoke("Chunk.retrieve",{key:e.key,source:e.source.rpcId})}copyChunkToGPU(e){++this.gpuMemoryGeneration;let t=this.rpc;if(e.state===z.SYSTEM_MEMORY)t.invoke("Chunk.update",{id:e.key,source:e.source.rpcId,state:z.GPU_MEMORY});else{let r={},n=[];e.serialize(r,n),r.state=z.GPU_MEMORY,t.invoke("Chunk.update",r,n)}}moveChunkToFrontend(e){let t=this.rpc,r={},n=[];e.serialize(r,n),r.state=z.SYSTEM_MEMORY,t.invoke("Chunk.update",r,n)}processQueuePromotions_(){let e=r=>{switch(r.state){case z.DOWNLOADING:fg(r);break;case z.GPU_MEMORY:this.freeChunkGPUMemory(r);case z.SYSTEM_MEMORY_WORKER:case z.SYSTEM_MEMORY:this.freeChunkSystemMemory(r);break}this.updateChunkState(r,z.QUEUED)},t=(r,n,i)=>{let s=this.systemMemoryEvictionQueue.candidates(),o=this.systemMemoryCapacity;for(;;){let a=r.next();if(a.done)return;let c=a.value,u=0,l=c.priorityTier,f=c.priority;if(!Hu(u,i,l,f,n,e)||!Hu(u,o,l,f,s,e))return;this.updateChunkState(c,z.DOWNLOADING),IC(c)}};for(let r=0;r<bC;++r)t(this.queuedDownloadPromotionQueue[r].candidates(),this.downloadEvictionQueue[r].candidates(),this.downloadCapacity[r]);t(this.queuedComputePromotionQueue.candidates(),this.computeEvictionQueue.candidates(),this.computeCapacity)}process(){if(!this.updatePending)return;this.updatePending=null;let e=this.gpuMemoryGeneration;this.processGPUPromotions_(),this.processQueuePromotions_(),this.logStatistics(),this.gpuMemoryGeneration!==e&&this.gpuMemoryChanged.dispatch()}logStatistics(){Ji&&console.log(`[Chunk status] QUEUED: ${this.numQueued}, FAILED: ${this.numFailed}, DOWNLOAD: ${this.downloadCapacity}, MEM: ${this.systemMemoryCapacity}, GPU: ${this.gpuMemoryCapacity}`)}invalidateSourceCache(e){for(let t of e.chunks.values()){switch(t.state){case z.DOWNLOADING:fg(t);break;case z.SYSTEM_MEMORY_WORKER:t.freeSystemMemory();break}this.updateChunkState(t,z.QUEUED)}this.rpc.invoke("Chunk.update",{source:e.rpcId}),this.scheduleUpdate()}};hg=Xu([G(vm)],hg);var Dr=class extends me{chunkManagerGeneration=-1;numVisibleChunksNeeded=0;numVisibleChunksAvailable=0;numPrefetchChunksNeeded=0;numPrefetchChunksAvailable=0},pg=200,dg=class extends me{queueManager;existingTierChunks=[];newTierChunks=[];updatePending=null;recomputeChunkPriorities=new $e;recomputeChunkPrioritiesLate=new $e;memoize=new kn;layers=[];sendLayerChunkStatistics=this.registerCancellable(Ws(()=>{this.rpc.invoke(Em,{id:this.rpcId,layers:this.layers.map(e=>({id:e.rpcId,numVisibleChunksAvailable:e.numVisibleChunksAvailable,numVisibleChunksNeeded:e.numVisibleChunksNeeded,numPrefetchChunksAvailable:e.numPrefetchChunksAvailable,numPrefetchChunksNeeded:e.numPrefetchChunksNeeded}))})},pg));constructor(e,t){super(e,t),this.queueManager=e.get(t.chunkQueueManager).addRef(),this.registerDisposer(this.queueManager.gpuMemoryChanged.add(this.registerCancellable(Ws(()=>this.scheduleUpdateChunkPriorities(),pg,{leading:!1,trailing:!0}))));for(let r=se.FIRST_TIER;r<=se.LAST_TIER;++r)r!==se.RECENT&&(this.existingTierChunks[r]=[])}scheduleUpdateChunkPriorities(){this.updatePending===null&&(this.updatePending=setTimeout(this.recomputeChunkPriorities_.bind(this),0))}registerLayer(e){let t=this.recomputeChunkPriorities.count;e.chunkManagerGeneration!==t&&(e.chunkManagerGeneration=t,this.layers.push(e),e.numVisibleChunksAvailable=0,e.numVisibleChunksNeeded=0,e.numPrefetchChunksAvailable=0,e.numPrefetchChunksNeeded=0)}recomputeChunkPriorities_(){this.updatePending=null,this.layers.length=0,this.recomputeChunkPriorities.dispatch(),this.recomputeChunkPrioritiesLate.dispatch(),this.updateQueueState([se.VISIBLE,se.PREFETCH]),this.sendLayerChunkStatistics()}requestChunk(e,t,r,n=z.GPU_MEMORY){if(Number.isNaN(r))return;if(t===se.RECENT)throw new Error("Not going to request a chunk with the RECENT tier");e.newRequestedState=Math.min(e.newRequestedState,n),e.newPriorityTier===se.RECENT&&this.newTierChunks.push(e);let i=e.newPriorityTier;(t<i||t===i&&r>e.newPriority)&&(e.newPriorityTier=t,e.newPriority=r)}updateQueueState(e){let t=this.existingTierChunks,r=this.queueManager;for(let i of e){let s=t[i];Ji&&console.log(`existingTierChunks[${se[i]}].length=${s.length}`);for(let o of s)o.newPriorityTier===se.RECENT&&r.performChunkPriorityUpdate(o);s.length=0}let n=this.newTierChunks;for(let i of n)r.performChunkPriorityUpdate(i),t[i.priorityTier].push(i);Ji&&console.log(`updateQueueState: newTierChunks.length = ${n.length}`),n.length=0,this.queueManager.scheduleUpdate()}};dg=Xu([G(xm)],dg);function ye(e,t){let r=class extends e{parameters;constructor(...n){super(...n);let i=n[1];this.parameters=i.parameters}};return r=Xu([hm(t.RPC_ID)],r),r}function Xe(e){return class extends e{chunkManager;constructor(...t){super(...t);let r=t[0],n=t[1];this.chunkManager=r.get(n.chunkManager)}}}X(Sm,function(e){let t=this.get(e.id);t.chunkManager.queueManager.invalidateSourceCache(t)});xt(wm,function(e){let t=this.get(e.queue),r=new Map;for(let n of t.sources)r.set(n.rpcId,n.statistics);return Promise.resolve({value:r})});var So=class extends be{};function Zu(e){let t,r;return async(n,i)=>((r===void 0||n!==void 0&&t?.generation===n.generation)&&(t=void 0,r=Yi(async s=>(t=await e(n,s),t))),r(i??{}))}var wo=class extends be{constructor(t){super(),this.base=t}memoize=new kn;getCredentialsProvider(t,r){return this.memoize.get({key:t,parameters:r},()=>this.registerDisposer(this.base.getCredentialsProvider(t,r).addRef()))}};var gg="CredentialsProvider",yg="CredentialsProvider.get",vg="CredentialsManager",xg="CredentialsManager.get";var MC=Object.defineProperty,TC=Object.getOwnPropertyDescriptor,Eg=(e,t,r,n)=>{for(var i=n>1?void 0:n?TC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&MC(t,r,i),i};var Sg=class extends me{get=Zu((e,t)=>this.rpc.promiseInvoke(yg,{providerId:this.rpcId,invalidCredentials:e},{signal:t.signal,progressListener:t.progressListener}))};Sg=Eg([G(gg)],Sg);function Nn(){return e=>class extends e{credentialsProvider;constructor(...t){super(...t);let r=t[1];this.credentialsProvider=this.rpc.getOptionalRef(r.credentialsProvider)}}}var Qu=class extends So{constructor(t,r,n,i){super(),this.rpc=t,this.managerId=r,this.key=n,this.parameters=i}get=Zu((t,r)=>this.rpc.promiseInvoke(xg,{managerId:this.managerId,key:this.key,parameters:this.parameters,invalidCredentials:t},{signal:r.signal,progressListener:r.progressListener}))},wg=class extends me{impl=new wo(this.makeBaseCredentialsManager());makeBaseCredentialsManager(){return{getCredentialsProvider:(e,t)=>new Qu(this.rpc,this.rpcId,e,t)}}getCredentialsProvider(e,t){return this.impl.getCredentialsProvider(e,t)}};wg=Eg([G(vg)],wg);function St(e,t){return e<t?-1:e>t?1:0}var Wi=class extends Error{constructor(t,r){super(`${t.getUrl()} not found`,r)}};async function Jt(e,t,r={}){return Pr(new Te(e,t),r)}async function Pr(e,t={}){let r=await e.read(t);if(t?.throwIfMissing===!0&&r===void 0)throw new Wi(e);if(t?.strictByteRange===!0&&r!==void 0){let{byteRange:n}=t,{offset:i,length:s}=r;if(n!==void 0&&("suffixLength"in n?s!==n.suffixLength:i!==n.offset||s!==void 0&&s!==n.length))throw new Error(`Received truncated response for ${e.getUrl()}, expected ${JSON.stringify(n)} but received offset=${i}, length=${s}`)}return r}function kC(e,t,r,n){switch(n){case"suffix":{let i=t.length;return{directories:e.directories.map(s=>s.substring(i)),entries:e.entries.map(({key:s,...o})=>({...o,key:s.substring(i)}))}}case"url":return{directories:e.directories.map(i=>r.getUrl(i)),entries:e.entries.map(({key:i,...s})=>({...s,key:r.getUrl(i)}))};default:return e}}async function Xi(e,t,r={}){if(!e.list)throw new Error("Listing not supported");return kC(await e.list(t,r),t,e,r.responseKeys)}var Te=class{constructor(t,r){this.store=t,this.key=r}stat(t){return this.store.stat(this.key,t)}read(t){return this.store.read(this.key,t)}getUrl(){return this.store.getUrl(this.key)}};function Eo(e){return e.entries.sort(({key:t},{key:r})=>St(t,r)),e.directories.sort(St),e}function NC(e){let t=e.match(/^((?:.*?\|)?)([a-zA-Z][a-zA-Z0-9-+.]*)(?:(:[^?#|]*)((?:[?#][^|]*)?))?$/);if(t===null)throw new Error(`Invalid URL: ${e}`);let[,r,n,i,s]=t;return i===void 0?`${r}${n}:`:i===":"||i.endsWith("/")?e:`${r}${n}${i}/${s??""}`}function bg(e){return e.match(/.*?([^|]*)$/)[1]}var DC=/^(?:([a-zA-Z][a-zA-Z0-9-+.]*):)?(.*)$/;function bo(e){let t=e.match(DC),r=t[1],n=t[2];return r===void 0?{url:e,scheme:e,suffix:void 0}:{url:e,scheme:r,suffix:n}}function Ig(e){return e.split("|").map(bo)}function Ve(e,...t){let[,r,n]=e.match(/^(.*?[^|?#]*)([^|]*)$/);for(let i of t)i.startsWith("/")&&(i=i.substring(1)),i!==""&&(r=NC(r),r+=Ze(i));return r+n}function el(e,...t){for(let r of t)r.startsWith("/")&&(r=r.substring(1)),r!==""&&(e=nn(e),e+=r);return e}function nn(e){return Ag(e)||(e+="/"),e}function Rr(e){let{suffix:t}=e;if(t!==void 0&&t.match(/[#?]/))throw new Error(`Invalid URL ${e.url}: query parameters and/or fragment not supported`)}function _g(e){if(e.suffix)throw new Error(`Invalid URL syntax ${JSON.stringify(e.url)}, expected "${e.scheme}:"`)}function PC(e){let[,t,r]=e.match(/^(.*?[^|?#]*)([^|]*)$/);return{base:t,queryAndFragment:r}}function Cg(e,t){let r=e;e.endsWith("/")&&(e=e.substring(0,e.length-1));for(let n of t.split("/"))if(!(n===""||n===".")){if(n===".."){let i=e.lastIndexOf("/");if(i<=0)throw new Error(`Invalid relative path ${JSON.stringify(t)} from base path ${JSON.stringify(r)}`);e=e.substring(0,i);continue}e!==""&&(e+="/"),e+=n}return t.endsWith("/")&&(e+="/"),e}function Ag(e){return e===""||e.endsWith("/")}function Ze(e){return encodeURI(e).replace(/[?#@]/g,t=>`%${t.charCodeAt(0).toString(16).toUpperCase()}`)}function Ht(e,t){let{base:r,queryAndFragment:n}=PC(e);return r+Ze(t)+n}function Dn(e){let t=new URL(e);if(t.hash)throw new Error("fragment not supported");if(t.username||t.password)throw new Error("basic auth credentials not supported");return{baseUrl:`${t.origin}/${t.search}`,path:decodeURIComponent(t.pathname.substring(1))}}function Mg(e){return async t=>{let r=[],n=await Promise.allSettled(e.map(i=>i.match(t)));for(let i of n)i.status==="fulfilled"&&r.push(...i.value);return r}}function RC(e){let t=new Set,r=new Set;for(let n of e){let{fileNames:i,subDirectories:s}=n;if(i!==void 0)for(let o of i)t.add(o);if(s!==void 0)for(let o of s)r.add(o)}return{fileNames:t,subDirectories:r,match:Mg(e)}}function OC(e){let t=0,r=0;for(let n of e)t=Math.max(t,n.prefixLength),r=Math.max(r,n.suffixLength);return{prefixLength:t,suffixLength:r,match:Mg(e)}}var Pn=class{directorySpecs=[];fileSpecs=[];_directorySpec;_fileSpec;registerDirectoryFormat(t){this.directorySpecs.push(t),this._directorySpec=void 0}registerFileFormat(t){this.fileSpecs.push(t),this._fileSpec=void 0}copyTo(t){t.directorySpecs.push(...this.directorySpecs),t.fileSpecs.push(...this.fileSpecs),t._fileSpec=void 0,t._directorySpec=void 0}get directorySpec(){return this._directorySpec??(this._directorySpec=this.getDirectorySpec())}getDirectorySpec(){return RC(this.directorySpecs)}get fileSpec(){return this._fileSpec??(this._fileSpec=this.getFileSpec())}getFileSpec(){let{fileSpecs:t}=this,r=[...t];return OC(r)}};var Io=class{baseKvStoreProviders=new Map;kvStoreAdapterProviders=new Map;autoDetectRegistry=new Pn;getKvStore(t){let r=Ig(t),n;{let i=r[0];n=this.getBaseKvStoreProvider(i).getKvStore(i)}for(let i=1;i<r.length;++i)n=this.applyKvStoreAdapterUrl(n,r[i]);return n}getFileHandle(t){let{store:r,path:n}=this.getKvStore(t);return new Te(r,n)}getBaseKvStoreProvider(t){let r=this.baseKvStoreProviders.get(t.scheme);if(r===void 0){let n=this.describeProtocolUsage(t.scheme),i=`Invalid base kvstore protocol "${t.scheme}:"`;throw n!==void 0&&(i+=`; ${n}`),new Error(i)}return r}getKvStoreAdapterProvider(t){let r=this.kvStoreAdapterProviders.get(t.scheme);if(r===void 0){let n=this.describeProtocolUsage(t.scheme),i=`Invalid kvstore adapter protocol "${t.scheme}:"`;throw n!==void 0&&(i+=`; ${n}`),i+=`; supported schemes: ${JSON.stringify(Array.from(this.kvStoreAdapterProviders.keys()))}`,new Error(i)}return r}applyKvStoreAdapterUrl(t,r){return this.getKvStoreAdapterProvider(r).getKvStore(r,t)}describeProtocolUsage(t){if(this.baseKvStoreProviders.has(t))return`"${t}:" may only be used as a base kvstore protocol`;if(this.kvStoreAdapterProviders.has(t))return`"${t}:" may only be used as a kvstore adapter protocol`}stat(t,r={}){let n=this.getKvStore(t);return n.store.stat(n.path,r)}read(t,r={}){let n=this.getKvStore(t);return Jt(n.store,n.path,r)}list(t,r={}){let n=this.getKvStore(t);return Xi(n.store,n.path,r)}resolveRelativePath(t,r){let n=this.getKvStore(t);return n.store.getUrl(Cg(n.path,r))}};var Zi=class{baseKvStoreProviders=[];kvStoreAdapterProviders=[];autoDetectRegistry=new Pn;registerBaseKvStoreProvider(t){this.baseKvStoreProviders.push(t)}registerKvStoreAdapterProvider(t){this.kvStoreAdapterProviders.push(t)}applyToContext(t){let{kvStoreContext:r}=t;for(let n of["baseKvStoreProviders","kvStoreAdapterProviders"]){let i=r[n];for(let s of this[n]){let o=s(t),{scheme:a}=o;if(i.has(a))throw new Error(`Duplicate kvstore scheme ${a}`);i.set(a,o)}}this.autoDetectRegistry.copyTo(t.kvStoreContext.autoDetectRegistry)}},Wt=new Zi;var Tg="SharedKvStoreContext",kg="SharedKvStoreContext.stat",Ng="SharedKvStoreContext.read",tl="SharedKvStoreContext.list",Dg="SharedKvStoreContext.completeUrl";var UC=Object.defineProperty,LC=Object.getOwnPropertyDescriptor,FC=(e,t,r,n)=>{for(var i=n>1?void 0:n?LC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&UC(t,r,i),i};var Pg=class extends me{kvStoreContext;chunkManager;credentialsManager;constructor(e,t){super(e,t),this.chunkManager=e.get(t.chunkManager),this.credentialsManager=e.get(t.credentialsManager),this.kvStoreContext=new Io,Wt.applyToContext(this),wt.applyToContext(this)}};Pg=FC([G(Tg)],Pg);var wt=new Zi;function ke(e){return class extends e{sharedKvStoreContext;constructor(...t){super(...t);let r=t[1];this.sharedKvStoreContext=this.rpc.get(r.sharedKvStoreContext)}}}var Rg="rendered_view.addLayer",Og="rendered_view.removeLayer",Ug="SharedProjectionParameters",Lg="SharedProjectionParameters.changed";var BC=Object.defineProperty,zC=Object.getOwnPropertyDescriptor,$C=(e,t,r,n)=>{for(var i=n>1?void 0:n?zC(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&BC(t,r,i),i};var rl=class extends be{constructor(t){super(),this.view=t}state=void 0},ir=class extends Dr{attachments=new Map;attach(t){}};X(Rg,function(e){let t=this.get(e.view),r=this.get(e.layer),n=new rl(t);r.attachments.set(t,n),r.attach(n)});X(Og,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);r.attachments.delete(t),n.dispose()});var Fg=class extends me{value;oldValue;changed=new Gt;constructor(e,t){super(e,t),this.value=t.value,this.oldValue=Object.assign({},this.value)}};Fg=$C([G(Ug)],Fg);X(Lg,function(e){let t=this.get(e.id),{value:r,oldValue:n}=t;Object.assign(n,r),Object.assign(r,e.value),t.changed.dispatch(n,r)});function VC(e,t,r){for(let n=0;n<r;++n){let i=t*n;e.fill(0,i,i+r),e[i+n]=1}return e}function nl(e,t,r=t){return VC(new e(t*r),t,Math.min(t,r))}function Bg(e,t,r,n,i,s){for(let o=0;o<s;++o){let a=o*n,c=o*t;for(let u=0;u<i;++u)e[c+u]=r[a+u]}return e}var kt;function GC(e,t,r){let n=1;(kt===void 0||kt.length<r)&&(kt=new Uint32Array(r));for(let i=0;i<r;++i)kt[i]=i;for(let i=0;i<r;++i){let s=t*i,o=i;{let u=Math.abs(e[s+i]);for(let l=i+1;l<r;++l){let f=Math.abs(e[s+l]);f>u&&(u=f,o=l)}}if(i!==o){n*=-1;for(let u=0;u<r;++u){let l=t*u,f=e[l+i];e[l+i]=e[l+o],e[l+o]=f}{let u=kt[i];kt[i]=kt[o],kt[o]=u}}let a=e[s+i],c=1/a;n*=a;for(let u=0;u<r;++u)e[t*u+i]*=c;e[s+i]=c;for(let u=0;u<r;++u){if(u===i)continue;let l=-e[t*i+u];for(let f=0;f<r;++f){let h=t*f;e[h+u]+=l*e[h+i]}e[t*i+u]=l*c}}for(let i=0;i<r;++i){let s=kt[i];for(;s!==i;){let o=t*i,a=t*s;for(let u=0;u<r;++u){let l=o+u,f=a+u,h=e[l];e[l]=e[f],e[f]=h}let c=kt[i]=kt[s];kt[s]=s,s=c}}return n}function il(e,t,r,n,i){return Bg(e,t,r,n,i,i),GC(e,t,i)}var jC=[{prefix:"Y",exponent:24,longPrefix:"yotta"},{prefix:"Z",exponent:21,longPrefix:"zetta"},{prefix:"E",exponent:18,longPrefix:"exa"},{prefix:"P",exponent:15,longPrefix:"peta"},{prefix:"T",exponent:12,longPrefix:"tera"},{prefix:"G",exponent:9,longPrefix:"giga"},{prefix:"M",exponent:6,longPrefix:"mega"},{prefix:"k",exponent:3,longPrefix:"kilo"},{prefix:"",exponent:0,longPrefix:""},{prefix:"m",exponent:-3,longPrefix:"milli"},{prefix:"\xB5",exponent:-6,longPrefix:"micro"},{prefix:"n",exponent:-9,longPrefix:"nano"},{prefix:"p",exponent:-12,longPrefix:"pico"},{prefix:"f",exponent:-15,longPrefix:"femto"},{prefix:"a",exponent:-18,longPrefix:"atto"},{prefix:"z",exponent:-21,longPrefix:"zepto"},{prefix:"y",exponent:-24,longPrefix:"yocto"}],sl=[...jC,{prefix:"h",exponent:2,longPrefix:"hecto"},{prefix:"da",exponent:1,longPrefix:"deca"},{prefix:"d",exponent:-1,longPrefix:"deci"},{prefix:"c",exponent:-2,longPrefix:"centi"}],qC=[{prefix:"u",exponent:-6},...sl],ol=new Map;ol.set("",{unit:"",exponent:0});var KC=new Map;for(let{prefix:e,exponent:t}of qC){KC.set(t,e);for(let r of["m","s","Hz","rad/s"])ol.set(`${e}${r}`,{unit:r,exponent:t})}function al(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=t[i]+r[i];return e}function zg(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=t[i]*r[i];return e}function Qi(e){let t=1;for(let r=0,n=e.length;r<n;++r)t*=e[r];return t}function cl(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=Math.min(t[i],r[i]);return e}function $g(e,t,r){let n=e.length;for(let i=0;i<n;++i)e[i]=Math.max(t[i],r[i]);return e}var Vg=new Float32Array(0),ul=new Float64Array(0),HO=Float64Array.of(1,1,1);function jg(e){let{names:t,units:r,scales:n}=e,{valid:i=!0,rank:s=t.length,timestamps:o=t.map(()=>Number.NEGATIVE_INFINITY),ids:a=t.map((f,h)=>-h),boundingBoxes:c=[]}=e,{coordinateArrays:u=new Array(s)}=e,{bounds:l=ZC(c,s)}=e;return{valid:i,rank:s,names:t,timestamps:o,ids:a,units:r,scales:n,boundingBoxes:c,bounds:l,coordinateArrays:u}}var WC=jg({valid:!1,names:[],units:[],scales:ul,boundingBoxes:[]}),qg=jg({valid:!0,names:[],units:[],scales:ul,boundingBoxes:[]});function XC(e,t,r){let{box:{lowerBounds:n,upperBounds:i},transform:s}=e,o=n.length,a=r,c=s[a*o+t],u=c,l=c,f=!1;for(let h=0;h<o;++h){let p=s[a*h+t];if(p===0)continue;let d=p*n[h],m=p*i[h];u+=Math.min(d,m),l+=Math.max(d,m),f=!0}if(f)return{lower:u,upper:l}}var _o=.001;function ZC(e,t){let r=new Float64Array(t),n=new Float64Array(t);r.fill(Number.NEGATIVE_INFINITY),n.fill(Number.POSITIVE_INFINITY);let i=new Array(t);i.fill(0);let s=new Array(t);s.fill(0);for(let a of e)for(let c=0;c<t;++c){let u=XC(a,c,t);if(u===void 0)continue;let{lower:l,upper:f}=u;if(Number.isFinite(l)&&Number.isFinite(f)){let h,p,d,m;Math.abs(l-(h=Math.round(l)))<_o&&Math.abs(f-(p=Math.round(f)))<_o?(++s[c],l=h,f=p):Math.abs(l-(d=Math.floor(l))-.5)<_o&&Math.abs(f-(m=Math.floor(f))-.5)<_o&&(++i[c],l=d+.5,f=m+.5)}r[c]=r[c]===Number.NEGATIVE_INFINITY?l:Math.min(r[c],l),n[c]=n[c]===Number.POSITIVE_INFINITY?f:Math.max(n[c],f)}let o=s.map((a,c)=>i[c]>0&&a===0);return{lowerBounds:r,upperBounds:n,voxelCenterAtIntegerCoordinates:o}}var PU=B.create(),RU=Qt.create();function eA(e,t){return qt(e.globalDimensionNames,t.globalDimensionNames)&&qt(e.displayDimensionIndices,t.displayDimensionIndices)&&qt(e.canonicalVoxelFactors,t.canonicalVoxelFactors)&&qt(e.voxelPhysicalScales,t.voxelPhysicalScales)&&e.canonicalVoxelPhysicalSize===t.canonicalVoxelPhysicalSize&&qt(e.displayDimensionUnits,t.displayDimensionUnits)&&qt(e.displayDimensionScales,t.displayDimensionScales)}function On(e,t){let r=e.displayDimensionRenderInfo;return r===t?!0:eA(r,t)?(e.displayDimensionRenderInfo=t,!0):!1}var w4={channelCoordinateSpace:qg,shape:new Uint32Array(0),numChannels:1,coordinates:new Uint32Array(0)};function Yg(e,t,r,n,i){let s=t.length,o=r.length,a=e.length,c=!0;for(let u=0;u<n;++u){let l=u,f=0;for(let h=0;h<s;++h)f+=i[l+h*n]*t[h];l+=s*n;for(let h=0;h<o;++h)f+=i[l+h*n]*r[h];f+=i[l+o*n],u<a?e[u]=f:(f<0||f>=1)&&(c=!1)}return c}function Jg(e,t,r){e.fill(0),e[15]=1;let n=!0,{displayDimensionIndices:i}=t,{globalToRenderLayerDimensions:s,modelToRenderLayerTransform:o}=r,a=r.rank;for(let c=0;c<3;++c){let u=i[c];if(u===-1){n=!1;continue}let l=s[u];if(l===-1){n=!1;continue}e[c+12]=o[l+a*(a+1)];for(let f=0;f<3;++f)e[c+4*f]=o[l+(a+1)*f]}if(!n){let{globalDimensionNames:c}=t,u=Array.from(i.filter(l=>l!==-1),l=>c[l]).join(",\xA0");throw new Error(`Transform from model dimensions (${r.modelDimensionNames.join(",\xA0")}) to display dimensions (${u}) does not have full rank`)}}var Un=class e{size;transform;invTransform;detTransform;finiteRank;constructor(t,r,n){this.size=B.clone(t),this.transform=ge.clone(r),this.finiteRank=n;let i=ge.create(),s=il(i,4,r,4,4);if(s===0)throw new Error("Transform is singular");this.invTransform=i,this.detTransform=s}toObject(){return{size:this.size,transform:this.transform,finiteRank:this.finiteRank}}static fromObject(t){return new e(t.size,t.transform,t.finiteRank)}globalToLocalSpatial(t,r){return B.transformMat4(t,r,this.invTransform)}localSpatialVectorToGlobal(t,r){return rg(t,r,this.transform)}globalToLocalNormal(t,r){return ng(t,r,this.transform)}};var W=(e=>(e[e.UINT8=0]="UINT8",e[e.INT8=1]="INT8",e[e.UINT16=2]="UINT16",e[e.INT16=3]="INT16",e[e.UINT32=4]="UINT32",e[e.INT32=5]="INT32",e[e.UINT64=6]="UINT64",e[e.FLOAT32=7]="FLOAT32",e))(W||{});var ut={0:1,1:1,2:2,3:2,4:4,5:4,6:8,7:4},ll={0:Uint8Array,1:Int8Array,2:Uint16Array,3:Int16Array,4:Uint32Array,5:Int32Array,6:BigUint64Array,7:Float32Array};function Co(e,t,r=0,n=t.byteLength){let i=ut[e];return new ll[e](t,r,n/i)}var Ur=!1,tA=!1,rA=ge.create();function nA(e,t){let r=0,n=Math.abs(e.detTransform),{transform:i,size:s}=e;for(let o=0;o<3;++o){let a=0;for(let u=0;u<3;++u)a+=t[u*4+2]*i[4*o+u];let c=s[o];r+=Math.abs(a)*c,n*=c}return n/r}function Hg(e,t,r){let{curPositionInChunks:n,fixedPositionWithinChunk:i}=e,{nonDisplayLowerClipBound:s,nonDisplayUpperClipBound:o}=e,{rank:a,chunkDataSize:c,lowerChunkBound:u,upperChunkBound:l}=e.source.spec;if(!Yg(n,t,r,e.layerRank,e.fixedLayerToChunkTransform))return!1;let f=.001;for(let h=0;h<a;++h){let p=n[h];if(p<s[h]-f||p>o[h]+f)return Ur&&console.log("excluding source",e,`because of chunkDim=${h}, sum=${p}`,s,o,e.fixedLayerToChunkTransform),!1;let d=c[h],m=n[h]=Math.min(l[h]-1,Math.max(u[h],Math.floor(p/d)));i[h]=p-m*d}return!0}function iA(e,t){let r=t.length,n=0;if(Ur&&console.log(t),r>1){let i=0;for(let s=0;s<r;++s){let o=t[s],{chunkLayout:a}=o,c=nA(a,e);Ur&&console.log(`chunksize = ${a.size}, sliceArea = ${c}`),c>i&&(i=c,n=s)}}return n}var Ln=new Un(B.create(),ge.create(),0);function sA(e,t){if(e.displayDimensionRenderInfo!==t.displayDimensionRenderInfo||e.pixelSize!==t.pixelSize)return!0;let{viewMatrix:r}=e,{viewMatrix:n}=t;for(let i=0;i<12;++i)if(r[i]!==n[i])return!0;return!1}var Ao=class extends rn{constructor(t){super(),this.projectionParameters=t,this.registerDisposer(t.changed.add((r,n)=>{sA(r,n)&&this.invalidateVisibleSources(),this.invalidateVisibleChunks()}))}visibleLayers=new Map;visibleSourcesStale=!0;invalidateVisibleSources(){this.visibleSourcesStale=!0}invalidateVisibleChunks(){}updateVisibleSources(){if(!this.visibleSourcesStale)return;this.visibleSourcesStale=!1;let t=this.projectionParameters.value.displayDimensionRenderInfo,{visibleLayers:r}=this;for(let[n,i]of r){let{allSources:s,visibleSources:o}=i;if(o.length=0,s.length===0||!On(i,t))continue;let a=iA(this.projectionParameters.value.viewMatrix,s.map(u=>u[0])),c=s[a];for(let u of n.filterVisibleSources(this,c))o.push(u);o.reverse(),Ur&&console.log("visible sources chosen",o)}}};function*Wg(e,t,r){let n=e.projectionParameters.value.pixelSize*1.1,i=r[0].effectiveVoxelSize,s=t.renderScaleTarget.value,o=l=>{let f=n*s;for(let h=0;h<3;++h){let p=l[h];if(p>f&&p>1.01*i[h])return!0}return!1},a=(l,f)=>{let h=n*s;for(let p=0;p<3;++p){let d=l[p],m=f[p];if(Math.abs(h-d)<Math.abs(h-m)&&d<1.01*m)return!0}return!1},c=r.length-1,u;for(Ur&&console.log(`Filtering ${r.length} visible sources`);;){let l=r[c];if(u!==void 0&&!a(l.effectiveVoxelSize,u)){Ur&&console.log(`  Stopping at ${c} because can't improve on prev voxel size: effectiveVoxelSize=${l.effectiveVoxelSize} prevVoxelSize=${u}`);break}if(yield l,c===0){Ur&&console.log("  Stopping because scaleIndex=0");break}if(!o(l.effectiveVoxelSize)){Ur&&console.log(`Stopping at at ${c} because can't improve on voxel size ${l.effectiveVoxelSize}`);break}u=l.effectiveVoxelSize,--c}}var Xg="SliceView",Zg="sliceview/RenderLayer",Qg="SliceView.addVisibleLayer",ey="SliceView.removeVisibleLayer",ty="ChunkManager.requestChunk",fl=new Float32Array(3),hl=new Float32Array(3),ry=ge.create(),ny=new Float32Array(24);function iy(e,t,r,n){let i=fl,s=hl,{lowerChunkDisplayBound:o,upperChunkDisplayBound:a}=t;for(let f=0;f<3;++f)i[f]=Math.max(i[f],o[f]),s[f]=Math.min(s[f],a[f]);let{curPositionInChunks:c,chunkDisplayDimensionIndices:u}=t;function l(){if(!n(i[0],i[1],i[2],s[0],s[1],s[2],e))return;let f=0,h=Math.max(0,s[0]-i[0]),p=h;for(let y=1;y<3;++y){let I=Math.max(0,s[y]-i[y]);p*=I,I>h&&(h=I,f=y)}if(p===0)return;if(p===1){c[u[0]]=i[0],c[u[1]]=i[1],c[u[2]]=i[2],r(i,e);return}let d=i[f],m=s[f],g=Math.floor(.5*(d+m));s[f]=g,l(),s[f]=m,i[f]=g,l(),i[f]=d}l()}function Mo(e,t,r,n){if(!Hg(r,e.globalPosition,t))return;let{size:i}=r.chunkLayout,s=ge.multiply(ry,e.viewProjectionMat,r.chunkLayout.transform);for(let u=0;u<3;++u){let l=i[u];for(let f=0;f<4;++f)s[4*u+f]*=l}let o=ny;ho(o,s);let a=fl,c=hl;a.fill(Number.NEGATIVE_INFINITY),c.fill(Number.POSITIVE_INFINITY),iy(o,r,n,po)}function To(e,t,r,n,i){if(!Hg(r,e.globalPosition,t))return;let{size:s}=n,o=ge.multiply(ry,e.viewProjectionMat,n.transform);for(let d=0;d<3;++d){let m=s[d];for(let g=0;g<4;++g)o[4*d+g]*=m}let{upperChunkDisplayBound:a}=r,c=rA;ge.invert(c,o);let u=fl,l=hl,f=1e-4,h=.001;for(let d=0;d<3;++d){let m=c[12+d]+f/s[d],g=Math.abs(c[d]),y=Math.abs(c[4+d]),I=a[d],_=m-g-y;_>=I&&_<I+h?_=I-1:_=Math.floor(_),u[d]=_,l[d]=Math.floor(m+g+y+1)}let p=ny;for(let d=0;d<3;++d){let m=o[4*d],g=o[4*d+1],y=o[4*d+2];p[d]=m,p[4+d]=-m,p[8+d]=+g,p[12+d]=-g,p[16+d]=+y,p[20+d]=-y}{let m=o[12],g=o[4*3+1],y=o[4*3+2];p[3]=1+m,p[7]=1-m,p[11]=1+g,p[15]=1-g,p[19]=y,p[23]=-y}tA&&(console.log("clippingPlanes",p),console.log("modelViewProjection",o.join(",")),console.log(`lower=${u.join(",")}, upper=${l.join(",")}`)),iy(p,r,i,sg)}function ko(e,t){let{finiteRank:r}=t;if(r===3)return t;Ln.finiteRank=r,B.copy(Ln.size,t.size);let n=ge.copy(Ln.transform,t.transform),i=ge.copy(Ln.invTransform,t.invTransform);Ln.detTransform=t.detTransform;let{invViewMatrix:s,width:o,height:a}=e,c=mo(e.projectionMat);for(let u=r;u<3;++u){let l=s[12+u],f=l,h=l,p=Math.abs(s[u]*o);f-=p,h+=p;let d=Math.abs(s[u+4]*a);f-=d,h+=d;let m=Math.abs(s[u+8]*c);f-=m,h+=m;let g=Math.max(1,h-f);n[12+u]=f,n[5*u]=g}return ge.invert(i,n),Ln}function sy(e){let t=.254829592,r=-.284496736,n=1.421413741,i=-1.453152027,s=1.061405429,a=1/(1+.3275911*Math.abs(e)),c=1-((((s*a+i)*a+n)*a+r)*a+t)*a*Math.exp(-e*e);return Math.sign(e)*c}var No=class{constructor(t=50,r=1e3){this.velocityHalfLifeMilliseconds=t,this.modelHalfLifeMilliseconds=r}lastTime=Number.NEGATIVE_INFINITY;rank=0;numSamples=0;prevPosition=new Float32Array;velocity=new Float32Array;mean=new Float32Array;variance=new Float32Array;reset(t){this.lastTime=Number.NEGATIVE_INFINITY,this.rank=t,this.numSamples=0,this.velocity=new Float32Array(t),this.prevPosition=new Float32Array(t),this.mean=new Float32Array(t),this.variance=new Float32Array(t)}addSample(t,r=Date.now()){let n=t.length;n!==this.rank&&this.reset(n);let i=this.numSamples;if(++this.numSamples,this.numSamples===0){this.prevPosition.set(t),this.lastTime=r;return}let s=r-this.lastTime;this.lastTime=r;let o=1-2**-(s/this.velocityHalfLifeMilliseconds),a=1-2**-(s/this.modelHalfLifeMilliseconds),{velocity:c,prevPosition:u,mean:l,variance:f}=this;for(let h=0;h<n;++h){let p=(t[h]-u[h])/Math.max(s,1);u[h]=t[h];let d=c[h],m=c[h]=d+o*(p-d);if(i===1)l[h]=m;else{let g=l[h],y=f[h],I=m-g;l[h]=g+a*I,f[h]=(1-a)*(y+a*I*I)}}}};function Et(e){return class extends e{visibility;constructor(...t){super(...t);let r=t[0],n=t[1];this.visibility=r.get(n.visibility),this.registerDisposer(this.visibility.changed.add(()=>this.chunkManager.scheduleUpdateChunkPriorities()))}}}function Ge(e){return e===Number.POSITIVE_INFINITY?se.VISIBLE:se.PREFETCH}function je(e){return e===Number.POSITIVE_INFINITY?0:e*ym}var oA=Object.defineProperty,aA=Object.getOwnPropertyDescriptor,py=(e,t,r,n)=>{for(var i=n>1?void 0:n?aA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&oA(t,r,i),i};var dl=-1e12,ts=1e9,oy=B.create(),cA=B.create(),uA=B.create(),pl=class extends Ao{constructor(t,r){super(t.get(r.projectionParameters)),this.initializeSharedObject(t,r.id)}};function ay(e){for(let t of e)for(let r of t)r.source.dispose()}var lA=Et(Xe(pl)),cy=class extends lA{velocityEstimator=new No;constructor(e,t){super(e,t),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateVisibleChunks()})),this.registerDisposer(this.projectionParameters.changed.add(()=>{this.velocityEstimator.addSample(this.projectionParameters.value.globalPosition)}))}invalidateVisibleChunks(){super.invalidateVisibleChunks(),this.chunkManager.scheduleUpdateChunkPriorities()}handleLayerChanged=()=>{this.chunkManager.scheduleUpdateChunkPriorities()};updateVisibleChunks(){let e=this.projectionParameters.value,t=this.chunkManager,r=this.visibility.value;if(r===Number.NEGATIVE_INFINITY)return;this.updateVisibleSources();let{centerDataPosition:n}=e,i=Ge(r),s=je(r);s+=dl;let o=cA,a=uA,c=[];this.velocityEstimator.addSample(this.projectionParameters.value.globalPosition);for(let[u,l]of this.visibleLayers){t.registerLayer(u);let{visibleSources:f}=l;for(let h=0,p=f.length;h<p;++h){let d=f[h],m=t.queueManager.enablePrefetch.value?hA(this.velocityEstimator,d):[],{chunkLayout:g}=d;g.globalToLocalSpatial(o,n);let{size:y,finiteRank:I}=g;B.copy(a,y);for(let C=I;C<3;++C)a[C]=0,o[C]=0;let E=s+ts*h;c.length=0;let b=mg();if(To(e,d.renderLayer.localPosition.value,d,ko(e,d.chunkLayout),C=>{B.multiply(oy,C,a);let v=-B.distance(o,oy),{curPositionInChunks:w}=d,x=d.source.getChunk(w);t.requestChunk(x,i,E+v),++u.numVisibleChunksNeeded,x.state===z.GPU_MEMORY&&++u.numVisibleChunksAvailable,c.push(x),x.markGeneration=b}),m.length!==0){let{curPositionInChunks:C}=d;for(let v of c){C.set(v.chunkGridPosition);for(let w=0,x=m.length;w<x;){let T=m[w],M=m[w+2],P=m[w+3],F=m[w+4],S=m[w+5],O=C[T],R=O+m[w+1];if(R<M||R>P){w=S;continue}C[T]=R;let N=d.source.getChunk(C);if(C[T]=O,N.markGeneration===b){w=S;continue}t.requestChunk(N,se.PREFETCH,E+F),++u.numPrefetchChunksNeeded,N.state===z.GPU_MEMORY&&++u.numPrefetchChunksAvailable,w+=es}}}}}}removeVisibleLayer(e){let{visibleLayers:t}=this,r=t.get(e);t.delete(e),ay(r.allSources),e.renderScaleTarget.changed.remove(this.invalidateVisibleSources),e.localPosition.changed.remove(this.handleLayerChanged),this.invalidateVisibleSources()}addVisibleLayer(e,t,r){let n=this.visibleLayers.get(e);n===void 0?(n={allSources:t,visibleSources:[],displayDimensionRenderInfo:r},this.visibleLayers.set(e,n),e.renderScaleTarget.changed.add(()=>this.invalidateVisibleSources()),e.localPosition.changed.add(this.handleLayerChanged)):(ay(n.allSources),n.allSources=t,n.visibleSources.length=0,n.displayDimensionRenderInfo=r),this.invalidateVisibleSources()}disposed(){for(let e of this.visibleLayers.keys())this.removeVisibleLayer(e);super.disposed()}invalidateVisibleSources(){super.invalidateVisibleSources(),this.chunkManager.scheduleUpdateChunkPriorities()}};cy=py([G(Xg)],cy);function sn(e,t,r){return t.map(i=>i.map(s=>{let o=e.getRef(s.source),a=s.chunkLayout,{rank:c}=o.spec;return{renderLayer:r,source:o,chunkLayout:Un.fromObject(a),layerRank:s.layerRank,nonDisplayLowerClipBound:s.nonDisplayLowerClipBound,nonDisplayUpperClipBound:s.nonDisplayUpperClipBound,lowerClipBound:s.lowerClipBound,upperClipBound:s.upperClipBound,lowerClipDisplayBound:s.lowerClipDisplayBound,upperClipDisplayBound:s.upperClipDisplayBound,lowerChunkDisplayBound:s.lowerChunkDisplayBound,upperChunkDisplayBound:s.upperChunkDisplayBound,effectiveVoxelSize:s.effectiveVoxelSize,chunkDisplayDimensionIndices:s.chunkDisplayDimensionIndices,fixedLayerToChunkTransform:s.fixedLayerToChunkTransform,combinedGlobalLocalToChunkTransform:s.combinedGlobalLocalToChunkTransform,curPositionInChunks:new Float32Array(c),fixedPositionWithinChunk:new Uint32Array(c)}}))}X(Qg,function(e){let t=this.get(e.id),r=this.get(e.layerId),n=sn(this,e.sources,r);t.addVisibleLayer(r,n,e.displayDimensionRenderInfo)});X(ey,function(e){let t=this.get(e.id),r=this.get(e.layerId);t.removeVisibleLayer(r)});var Fn=class extends fe{chunkGridPosition;source=null;initializeVolumeChunk(t,r){super.initialize(t),this.chunkGridPosition=Float32Array.from(r)}serialize(t,r){super.serialize(t,r),t.chunkGridPosition=this.chunkGridPosition}downloadSucceeded(){super.downloadSucceeded()}freeSystemMemory(){}toString(){return this.source.toString()+":"+tr(this.chunkGridPosition)}},Bn=class extends Ne{spec;constructor(t,r){super(t,r),this.spec=r.spec}getChunk(t){let r=t.join(),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(this.chunkConstructor),n.initializeVolumeChunk(r,t),this.addChunk(n)),n}},uy=class extends me{renderScaleTarget;localPosition;numVisibleChunksNeeded;numVisibleChunksAvailable;numPrefetchChunksNeeded;numPrefetchChunksAvailable;chunkManagerGeneration;constructor(e,t){super(e,t),this.renderScaleTarget=e.get(t.renderScaleTarget),this.localPosition=e.get(t.localPosition),this.numVisibleChunksNeeded=0,this.numVisibleChunksAvailable=0,this.numPrefetchChunksAvailable=0,this.numPrefetchChunksNeeded=0,this.chunkManagerGeneration=-1}filterVisibleSources(e,t){return Wg(e,this,t)}};uy=py([G(Zg)],uy);var ly=2e3,fA=.1,fy=32,hy=.05,es=6;function hA(e,t){let r=[],n=e.rank,{combinedGlobalLocalToChunkTransform:i,layerRank:s}=t,{rank:o,chunkDataSize:a}=t.source.spec,{mean:c,variance:u}=e;for(let l=0;l<o;++l){let f=t.chunkDisplayDimensionIndices.includes(l),h=0,p=0;for(let w=0;w<n;++w){let x=c[w],T=u[w],M=i[w*s+l];h+=M*x,p+=M*M*T}if(h>fA)continue;let d=a[l],m=f?0:t.fixedPositionWithinChunk[l]/d,g=h/d*ly,y=Math.sqrt(2*p)/d*ly;if(Math.abs(g)<.001&&y<.001)continue;y=Math.max(1e-6,y);let I=w=>.5*(1+sy((w-g)/y)),_=t.curPositionInChunks[l],E=Math.floor(t.lowerClipBound[l]/d),b=Math.ceil(t.upperClipBound[l]/d)-1,C=r.length;for(let w=1;w<=fy&&!(!f&&_+w>b);++w){let x=1-I(w-m);if(x<hy)break;r.push(l,w,E,b,x,0)}let v=r.length;for(let w=C,x=r.length;w<x;w+=es)r[w+es-1]=v;C=v;for(let w=1;w<=fy&&!(!f&&_-w<E);++w){let x=I(-w+1-m);if(x<hy)break;r.push(l,-w,E,b,x,0)}v=r.length;for(let w=C,x=r.length;w<x;w+=es)r[w+es-1]=v}return r}xt(ty,async function(e,t){let r=this.get(e.source),{chunkManager:n}=r,i=r.getChunk(e.chunkGridPosition),s=i.key;if(i.state<=z.SYSTEM_MEMORY)return{value:void 0};if(i.state===z.FAILED)throw i.error;let o=n.recomputeChunkPriorities.add(()=>{n.requestChunk(i,se.VISIBLE,Number.POSITIVE_INFINITY,z.SYSTEM_MEMORY)});n.scheduleUpdateChunkPriorities();let a,c=new Promise((u,l)=>{a=f=>{if(f.state===z.FAILED){l(f.error);return}f.state<=z.SYSTEM_MEMORY&&u()}});r.registerChunkListener(s,a);try{return await Qs(c,t.signal),{value:void 0}}finally{r.unregisterChunkListener(s,a),o(),n.scheduleUpdateChunkPriorities()}});var dy="perspective_view/PerspectiveView";var pA=Object.defineProperty,dA=Object.getOwnPropertyDescriptor,mA=(e,t,r,n)=>{for(var i=n>1?void 0:n?dA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&pA(t,r,i),i};var my=class extends me{visibility;projectionParameters;constructor(...e){super(...e);let t=e[0],r=e[1];this.visibility=t.get(r.visibility),this.projectionParameters=t.get(r.projectionParameters)}};my=mA([G(dy)],my);var rs=class extends ir{};var gy="volume_rendering/VolumeRenderingRenderLayer",yy="volume_rendering/VolumeRenderingRenderLayer/update",gA=!1,yA=Tt.create();function vy(e,t,r,n,i,s){if(n.length===0)return;let{viewMatrix:o,projectionMat:a,displayDimensionRenderInfo:c}=e,{voxelPhysicalScales:u}=c,l=Ki(u),f=mo(a),p=(f/r)**3,d=Tt.determinant(fo(yA,o)),m={spatialScales:new Map,activeIndex:-1},g=v=>{let w=n[v];return Math.abs(w.chunkLayout.detTransform*d)},y=n.length-1,I=g(y);for(let v=y;v>=0;--v){let w=g(v),x=Math.cbrt(w*l/d),T=f/Math.cbrt(w);m.spatialScales.set(x,T),w-p>=0&&(I=w,y=v),m.activeIndex=y}if(gA){console.log(n);for(let v=0;v<n.length;++v){let w=g(v),x=f/Math.cbrt(w);console.log(`scaleIndex=${v} viewVolume=${w} bestScaleIndex=${y} actualViewVolume=${p}, desiredSamples=${x}, difference=${w-p}`)}}let _=Math.cbrt(I*l/d),E=f/Math.cbrt(I),b=!0,C=n[y];Mo(e,t,C,(v,w)=>{b&&(i(C,y,_,E,w,m),b=!1),s(C,y,v)})}var vA=Object.defineProperty,xA=Object.getOwnPropertyDescriptor,SA=(e,t,r,n)=>{for(var i=n>1?void 0:n?xA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&vA(t,r,i),i};var xy=B.create(),wA=B.create(),EA=B.create(),bA=B.create(),Sy=class extends Xe(ir){localPosition;renderScaleTarget;constructor(e,t){super(e,t),this.renderScaleTarget=e.get(t.renderScaleTarget),this.localPosition=e.get(t.localPosition);let r=()=>this.chunkManager.scheduleUpdateChunkPriorities();this.registerDisposer(this.localPosition.changed.add(r)),this.registerDisposer(this.renderScaleTarget.changed.add(r)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(t),e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.state={displayDimensionRenderInfo:r.projectionParameters.value.displayDimensionRenderInfo,transformedSources:[]}}recomputeChunkPriorities(){for(let e of this.attachments.values()){let{view:t}=e,r=t.visibility.value;if(r===Number.NEGATIVE_INFINITY)continue;let n=e.state,{transformedSources:i}=n;if(i.length===0||!On(n,t.projectionParameters.value.displayDimensionRenderInfo))continue;let s=t.projectionParameters.value,o=Ge(r),a=je(r);a+=dl;let c=wA,u=EA,l=bA,{globalPosition:f,displayDimensionRenderInfo:{displayDimensionIndices:h}}=s;for(let m=0;m<3;++m){let g=h[m];l[m]=g===-1?0:f[g]}let p,{chunkManager:d}=this;d.registerLayer(this),vy(s,this.localPosition.value,this.renderScaleTarget.value,i[0],(m,g)=>{let{chunkLayout:y}=m;y.globalToLocalSpatial(c,l);let{size:I,finiteRank:_}=y;B.copy(u,I);for(let b=_;b<3;++b)u[b]=0,c[b]=0;let E=i[0].length-1-g;p=a+ts*E},(m,g,y)=>{B.multiply(xy,y,u);let I=-B.distance(c,xy),_=m.source.getChunk(m.curPositionInChunks);++this.numVisibleChunksNeeded,d.requestChunk(_,o,p+I),_.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable})}}};Sy=SA([G(gy)],Sy);X(yy,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);n.state.transformedSources=sn(this,e.sources,r),n.state.displayDimensionRenderInfo=e.displayDimensionRenderInfo,r.chunkManager.scheduleUpdateChunkPriorities()});var wy="annotation.MetadataChunkSource";var Ey="annotation.SubsetGeometryChunkSource",by="annotation.reference.add",Iy="annotation.reference.delete",_y="annotation.commit",ml="annotation.commit",Cy="annotation/SpatiallyIndexedRenderLayer",Ay="annotation/PerspectiveRenderLayer:updateSources",My="annotation/RenderLayer",Ty="annotation/RenderLayer.updateSegmentation",IA=Tt.create();function ky(e,t,r,n,i,s){let{displayDimensionRenderInfo:o,viewMatrix:a,projectionMat:c,width:u,height:l}=e,{voxelPhysicalScales:f}=o,h=Math.abs(Tt.determinant(fo(IA,a))),p=Ki(f),d=og(c)/h*p;if(n.length===0)return;let m=n[0],g=Math.abs(m.chunkLayout.detTransform)*p,{lowerClipDisplayBound:y,upperClipDisplayBound:I}=m;for(let w=0;w<3;++w)g*=I[w]-y[w];let _=Math.min(g,d),E=u*l,C=E/r**2/_,v=0;for(let w=n.length-1;w>=0&&v<C;--w){let x=n[w],T=x.source.spec,{chunkLayout:M}=x,P=Ki(M.size)*Math.abs(M.detTransform)*p,{limit:F,rank:S}=T,{nonDisplayLowerClipBound:O,nonDisplayUpperClipBound:R}=x,N=1;for(let ie=0;ie<S;++ie){let Ce=R[ie]-O[ie];Number.isFinite(Ce)&&(N/=Ce)}let U=F*N/P,L=!0,j=v+U,V=(1/j)**(1/3),Y=Math.sqrt(E/(j*_)),J=(C-v)*P/N,de=Math.min(1,J/T.limit);Mo(e,t,x,()=>{L&&(i(x,w),L=!1),s(x,w,de,V,Y)}),v=j}}var Lr=(e=>(e[e.MIN_REPRESENTATIVE=0]="MIN_REPRESENTATIVE",e[e.MAX_REPRESENTATIVE=1]="MAX_REPRESENTATIVE",e[e.REPRESENTATIVE_EXCLUDED=2]="REPRESENTATIVE_EXCLUDED",e[e.NONREPRESENTATIVE_EXCLUDED=4]="NONREPRESENTATIVE_EXCLUDED",e))(Lr||{});var yl=class{constructor(t){this.value=t,this.min=t}rank=0;parent=this;next=this;prev=this;min};function gl(e){let t=e,r=e.parent;for(;r!==e;)e=r,r=e.parent;for(e=t.parent;r!==e;)t.parent=r,t=e,e=t.parent;return r}function _A(e,t){let r=e.rank,n=t.rank;return r>n?(t.parent=e,e):(e.parent=t,r===n&&(t.rank=n+1),t)}function CA(e,t){let r=e.prev,n=t.prev;t.prev=r,r.next=t,e.prev=n,n.next=e}function*Ny(e){let t=e;do yield t.value,t=t.next;while(t!==e)}function Dy(e){return e.parent===e}var Do=class{map=new Map;visibleSegmentEquivalencePolicy=new jt(Lr.MIN_REPRESENTATIVE);generation=0;has(t){return this.map.has(t)}get(t){let r=this.map.get(t);return r===void 0?t:gl(r).min}isMinElement(t){return t===this.get(t)}makeSet(t){let{map:r}=this,n=r.get(t);return n===void 0?(n=new yl(t),r.set(t,n),n):gl(n)}link(t,r){let n=this.makeSet(t),i=this.makeSet(r);if(n===i)return!1;this.generation++;let s=_A(n,i);CA(n,i);let o=n.min,a=i.min,c=(this.visibleSegmentEquivalencePolicy.value&Lr.MAX_REPRESENTATIVE)!==0;return s.min=o<a===c?a:o,!0}linkAll(t){for(let r=1,n=t.length;r<n;++r)this.link(t[0],t[r])}deleteSet(t){let{map:r}=this,n=!1;for(let i of this.setElements(t))r.delete(i),n=!0;return n&&++this.generation,n}*setElements(t){let r=this.map.get(t);r===void 0?yield t:yield*Ny(r)}clear(){let{map:t}=this;return t.size===0?!1:(++this.generation,t.clear(),!0)}get size(){return this.map.size}*mappings(){for(let t of this.map.values())yield[t.value,gl(t).min]}*roots(){for(let t of this.map.values())Dy(t)&&(yield t.value)}[Symbol.iterator](){return this.mappings()}toJSON(){let t=new Array;for(let r of this.map.values())if(Dy(r)){let n=new Array;for(let i of Ny(r))n.push(i);n.sort(mr),t.push(n)}return t.sort((r,n)=>mr(r[0],n[0])),t.map(r=>r.map(n=>n.toString()))}};var AA=Object.defineProperty,MA=Object.getOwnPropertyDescriptor,TA=(e,t,r,n)=>{for(var i=n>1?void 0:n?MA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&AA(t,r,i),i};var kA="DisjointUint64Sets",Ry="DisjointUint64Sets.add",Oy="DisjointUint64Sets.clear",Uy="DisjointUint64Sets.highBitRepresentativeChanged",Ly="DisjointUint64Sets.deleteSet",Po=class extends me{disjointSets=new Do;changed=new $e;get value(){return this}static makeWithCounterpart(e,t){let r=new Po;return r.disjointSets.visibleSegmentEquivalencePolicy=t,r.registerDisposer(t.changed.add(()=>{Py(r)})),r.initializeCounterpart(e),t.value&&Py(r),r}link(e,t){if(this.disjointSets.link(e,t)){let{rpc:r}=this;return r&&r.invoke(Ry,{id:this.rpcId,a:e,b:t}),this.changed.dispatch(),!0}return!1}linkAll(e){for(let t=1,r=e.length;t<r;++t)this.link(e[0],e[t])}has(e){return this.disjointSets.has(e)}get(e){return this.disjointSets.get(e)}clear(){if(this.disjointSets.clear()){let{rpc:e}=this;e&&e.invoke(Oy,{id:this.rpcId}),this.changed.dispatch()}}setElements(e){return this.disjointSets.setElements(e)}deleteSet(e){if(this.disjointSets.deleteSet(e)){let{rpc:t}=this;t&&t.invoke(Ly,{id:this.rpcId,x:e}),this.changed.dispatch()}}get size(){return this.disjointSets.size}toJSON(){return this.disjointSets.toJSON()}restoreState(e){e!==void 0&&ct(e,t=>{let r;ct(t,n=>{let i=yr(n);r!==void 0&&this.link(r,i),r=i})})}assignFrom(e){this.clear(),e instanceof Po&&(e=e.disjointSets);for(let[t,r]of e)this.link(t,r)}};Po=TA([G(kA)],Po);X(Ry,function(e){let t=this.get(e.id);t.disjointSets.link(e.a,e.b)&&t.changed.dispatch()});X(Oy,function(e){let t=this.get(e.id);t.disjointSets.clear()&&t.changed.dispatch()});function Py(e){e.rpc.invoke(Uy,{id:e.rpcId,value:e.disjointSets.visibleSegmentEquivalencePolicy.value})}X(Uy,function(e){let t=this.get(e.id);t.disjointSets.visibleSegmentEquivalencePolicy.value=e.value});X(Ly,function(e){let t=this.get(e.id);t.disjointSets.deleteSet(e.x)&&t.changed.dispatch()});function zn(e,t){return t>>>=0,e>>>=0,t=Math.imul(t,3432918353)>>>0,t=(t<<15|t>>>17)>>>0,t=Math.imul(t,461845907)>>>0,e=(e^t)>>>0,e=(e<<13|e>>>19)>>>0,e=e*5+3864292196>>>0,e}function vl(e=128){let t=Math.ceil(e/32),r=new Uint32Array(t);crypto.getRandomValues(r);let n="";for(let i=0;i<t;++i)n+=("00000000"+r[i].toString(16)).slice(-8);return n}function Fy(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength),r=65536;for(let n=0,i=t.length;n<i;n+=r)crypto.getRandomValues(t.subarray(n,Math.min(i,n+r)));return e}var By=3,NA=.8,Fr=!1,sr=0n,zy=0n,Ro=class e{constructor(t=e.generateHashSeeds(By)){this.hashSeeds=t;let r=8;for(;r<2*t.length;)r*=2;this.allocate(r)}loadFactor=NA;size=0;table;tableSize;empty=0xffffffffffffffffn;maxRehashAttempts=5;maxAttempts=5;capacity;generation=0;mungedEmptyKey;updateHashFunctions(t){this.hashSeeds=e.generateHashSeeds(t),this.mungedEmptyKey=void 0}tableWithMungedEmptyKey(t){let r=this.hashSeeds.length,n=new Array(r);for(let a=0;a<r;++a)n[a]=this.getHash(a,this.empty);let{mungedEmptyKey:i}=this;if(i===void 0)e:for(;;){i=Nu();for(let a=0;a<r;++a){let c=this.getHash(a,i);for(let u=0;u<r;++u)if(n[u]===c)continue e}this.mungedEmptyKey=i;break}let{table:s,empty:o}=this;for(let a=0;a<r;++a){let c=n[a];s[c]===o&&(s[c]=i)}try{t(s)}finally{for(let a=0;a<r;++a){let c=n[a];s[c]===i&&(s[c]=o)}}}static generateHashSeeds(t=By){return Fy(new Uint32Array(t))}getHash(t,r){let n=this.hashSeeds[t];return n=zn(n,Number(r&0xffffffffn)),n=zn(n,Number(r>>32n)),this.entryStride*(n&this.tableSize-1)}*keys(){let{empty:t,entryStride:r}=this,{table:n}=this;for(let i=0,s=n.length;i<s;i+=r){let o=n[i];o!==t&&(yield o)}}indexOf(t){let{table:r,empty:n}=this;if(t===n)return-1;for(let i=0,s=this.hashSeeds.length;i<s;++i){let o=this.getHash(i,t);if(r[o]===t)return o}return-1}chooseAnotherEmptyKey(){let{empty:t,table:r,entryStride:n}=this,i;for(;i=Nu(),!(i!==t&&!this.has(i)););this.empty=i;for(let s=0,o=r.length;s<o;s+=n)r[s]===t&&(r[s]=i)}has(t){return this.indexOf(t)!==-1}delete(t){let r=this.indexOf(t);if(r!==-1){let{table:n}=this;return n[r]=this.empty,++this.generation,this.size--,!0}return!1}clearTable(){let{table:t,empty:r}=this;t.fill(r)}clear(){return this.size===0?!1:(this.size=0,++this.generation,this.clearTable(),!0)}reserve(t){return t>this.capacity?(this.backupPending(),this.grow(t),this.restorePending(),!0):!1}swapPending(t,r){let n=sr;this.storePending(t,r),t[r]=n}storePending(t,r){sr=t[r]}backupPending(){zy=sr}restorePending(){sr=zy}tryToInsert(){Fr&&console.log(`tryToInsert: ${sr}`);let t=0,{empty:r,maxAttempts:n,table:i}=this,s=this.hashSeeds.length,o=Math.floor(Math.random()*s);for(;;){let a=this.getHash(o,sr);if(this.swapPending(i,a),sr===r)return!0;if(++t===n)break;o=(o+Math.floor(Math.random()*(s-1))+1)%s}return!1}allocate(t){this.tableSize=t;let{entryStride:r}=this;this.table=new BigUint64Array(t*r),this.maxAttempts=t,this.clearTable(),this.capacity=t*this.loadFactor,this.mungedEmptyKey=void 0}rehash(t,r){Fr&&console.log("rehash begin"),this.allocate(r),this.updateHashFunctions(this.hashSeeds.length);let{empty:n,entryStride:i}=this;for(let s=0,o=t.length;s<o;s+=i)if(t[s]!==n&&(this.storePending(t,s),!this.tryToInsert()))return Fr&&console.log("rehash failed"),!1;return Fr&&console.log("rehash end"),!0}grow(t){Fr&&console.log(`grow: ${t}`);let r=this.table,{tableSize:n}=this;for(;n<t;)n*=2;for(;;){for(let i=0;i<this.maxRehashAttempts;++i)if(this.rehash(r,n)){Fr&&console.log("grow end");return}n*=2}}insertInternal(){for(++this.generation,sr===this.empty&&this.chooseAnotherEmptyKey(),++this.size>this.capacity&&(this.backupPending(),this.grow(this.tableSize*2),this.restorePending());!this.tryToInsert();)this.backupPending(),this.grow(this.tableSize),this.restorePending()}},is=class extends Ro{add(t){return this.has(t)?!1:(Fr&&console.log(`add: ${t}`),sr=t,this.insertInternal(),!0)}[Symbol.iterator](){return this.keys()}};is.prototype.entryStride=1;var ns=0n,$y=0n,ss=class extends Ro{set(t,r){return this.has(t)?!1:(Fr&&console.log(`add: ${t} -> ${r}`),sr=t,ns=r,this.insertInternal(),!0)}get(t){let r=this.indexOf(t);if(r!==-1)return this.table[r+1]}swapPending(t,r){let n=ns;super.swapPending(t,r),t[r+1]=n}storePending(t,r){super.storePending(t,r),ns=t[r+1]}backupPending(){super.backupPending(),$y=ns}restorePending(){super.restorePending(),ns=$y}[Symbol.iterator](){return this.entries()}*entries(){let{empty:t,entryStride:r}=this,{table:n}=this;for(let i=0,s=n.length;i<s;i+=r){let o=n[i];if(o!==t){let a=n[i+1];yield[o,a]}}}};ss.prototype.entryStride=2;var DA=Object.defineProperty,PA=Object.getOwnPropertyDescriptor,RA=(e,t,r,n)=>{for(var i=n>1?void 0:n?PA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&DA(t,r,i),i};var xl=class extends me{hashTable=new ss;changed=new Gt;get value(){return this}static makeWithCounterpart(e){let t=new xl;return t.initializeCounterpart(e),t}set_(e,t){return this.hashTable.set(e,t)}set(e,t){if(this.set_(e,t)){let{rpc:r}=this;r&&r.invoke("Uint64Map.set",{id:this.rpcId,key:e,value:t}),this.changed.dispatch(e,!0)}}has(e){return this.hashTable.has(e)}get(e){return this.hashTable.get(e)}[Symbol.iterator](){return this.hashTable.entries()}delete_(e){return this.hashTable.delete(e)}delete(e){if(this.delete_(e)){let{rpc:t}=this;t&&t.invoke("Uint64Map.delete",{id:this.rpcId,key:e}),this.changed.dispatch(e,!1)}}get size(){return this.hashTable.size}assignFrom(e){this.clear();for(let[t,r]of e)this.set(t,r)}clear(){if(this.hashTable.clear()){let{rpc:e}=this;e&&e.invoke("Uint64Map.clear",{id:this.rpcId}),this.changed.dispatch(null,!1)}}toJSON(){let e={};for(let[t,r]of this.hashTable.entries())e[t.toString()]=r.toString();return e}};xl=RA([G("Uint64Map")],xl);X("Uint64Map.set",function(e){let t=this.get(e.id);t.set_(e.key,e.value)&&t.changed.dispatch()});X("Uint64Map.delete",function(e){let t=this.get(e.id);t.delete_(e.key)&&t.changed.dispatch()});X("Uint64Map.clear",function(e){let t=this.get(e.id);t.hashTable.clear()&&t.changed.dispatch()});var OA=Object.defineProperty,UA=Object.getOwnPropertyDescriptor,LA=(e,t,r,n)=>{for(var i=n>1?void 0:n?UA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&OA(t,r,i),i};var $n=class extends me{hashTable=new is;changed=new Gt;get value(){return this}static makeWithCounterpart(e){let t=new $n;return t.initializeCounterpart(e),t}set(e,t){t?this.add(e):this.delete(e)}reserve_(e){return this.hashTable.reserve(e)}reserve(e){if(this.reserve_(e)){let{rpc:t}=this;t&&t.invoke("Uint64Set.reserve",{id:this.rpcId,value:e})}}add_(e){let t=!1;for(let r of e)t=this.hashTable.add(r)||t;return t}add(e){let t=typeof e=="bigint"?[e]:e;if(this.add_(t)){let{rpc:r}=this;r&&r.invoke("Uint64Set.add",{id:this.rpcId,value:t}),this.changed.dispatch(e,!0)}}has(e){return this.hashTable.has(e)}[Symbol.iterator](){return this.hashTable.keys()}keys(){return this.hashTable.keys()}delete_(e){let t=!1;for(let r of e)t=this.hashTable.delete(r)||t;return t}delete(e){let t=typeof e=="bigint"?[e]:e;if(this.delete_(t)){let{rpc:r}=this;r&&r.invoke("Uint64Set.delete",{id:this.rpcId,value:t}),this.changed.dispatch(e,!1)}}get size(){return this.hashTable.size}clear(){if(this.hashTable.clear()){let{rpc:e}=this;e&&e.invoke("Uint64Set.clear",{id:this.rpcId}),this.changed.dispatch(null,!1)}}toJSON(){let e=new Array;for(let t of this.keys())e.push(t.toString());return e.sort(),e}assignFrom(e){this.clear();for(let t of e.keys())this.add(t)}};$n=LA([G("Uint64Set")],$n);X("Uint64Set.reserve",function(e){let t=this.get(e.id);t.reserve_(e.value)&&t.changed.dispatch()});X("Uint64Set.add",function(e){let t=this.get(e.id);t.add_(e.value)&&t.changed.dispatch()});X("Uint64Set.delete",function(e){let t=this.get(e.id);t.delete_(e.value)&&t.changed.dispatch()});X("Uint64Set.clear",function(e){let t=this.get(e.id);t.hashTable.clear()&&t.changed.dispatch()});var Vy=["visibleSegments","segmentEquivalences","temporaryVisibleSegments","temporarySegmentEquivalences","useTemporaryVisibleSegments","useTemporarySegmentEquivalences"];function Oo(e,t,r){e.registerDisposer(t.visibleSegments.changed.add(r)),e.registerDisposer(t.segmentEquivalences.changed.add(r))}function Uo(e,t,r){e.registerDisposer(t.temporaryVisibleSegments.changed.add(r)),e.registerDisposer(t.temporarySegmentEquivalences.changed.add(r)),e.registerDisposer(t.useTemporaryVisibleSegments.changed.add(r)),e.registerDisposer(t.useTemporarySegmentEquivalences.changed.add(r))}function on(e){return e.toString()}function FA(e){return(e&0x8000000000000000n)!==0n}function BA(e){return e.useTemporaryVisibleSegments.value?e.temporaryVisibleSegments:e.visibleSegments}function zA(e){return e.useTemporarySegmentEquivalences.value?e.temporarySegmentEquivalences:e.segmentEquivalences}function xr(e,t){let r=BA(e),n=zA(e),i=n.disjointSets.visibleSegmentEquivalencePolicy.value;for(let s of r.keys())if(i&Lr.NONREPRESENTATIVE_EXCLUDED){let o=n.get(s);t(s,o)}else{if(!n.disjointSets.isMinElement(s))continue;for(let o of n.setElements(s))i&Lr.REPRESENTATIVE_EXCLUDED&&i&Lr.MAX_REPRESENTATIVE&&FA(o)||t(o,s)}}function Sl(e,t,r={}){for(let n of Vy)r[n]=e.get(t[n]);return r}var an=e=>class extends e{visibleSegments;selectedSegments;segmentEquivalences;temporaryVisibleSegments;temporarySegmentEquivalences;useTemporaryVisibleSegments;useTemporarySegmentEquivalences;transform;renderScaleTarget;constructor(...r){let[n,i]=r;super(n,i),Sl(n,i,this),this.transform=n.get(i.transform),this.renderScaleTarget=n.get(i.renderScaleTarget);let s=()=>{this.chunkManager.scheduleUpdateChunkPriorities()};Uo(this,this,s),Oo(this,this,s),this.registerDisposer(this.transform.changed.add(s)),this.registerDisposer(this.renderScaleTarget.changed.add(s))}};var $A=Object.defineProperty,VA=Object.getOwnPropertyDescriptor,Lo=(e,t,r,n)=>{for(var i=n>1?void 0:n?VA(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&$A(t,r,i),i};var GA=200,jA=60,wl=class extends fe{annotation;freeSystemMemory(){this.annotation=void 0}serialize(t,r){super.serialize(t,r),t.annotation=this.annotation}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=0,super.downloadSucceeded()}},Vn=class{data;typeToOffset;typeToIds;typeToIdMaps;typeToInstanceCounts;typeToSize;serialize(t,r){t.data=this.data,t.typeToOffset=this.typeToOffset,t.typeToIds=this.typeToIds,t.typeToIdMaps=this.typeToIdMaps,t.typeToInstanceCounts=this.typeToInstanceCounts,t.typeToSize=this.typeToSize,r.push(this.data.buffer)}get numBytes(){return this.data.byteLength}};function Yy(e){class t extends e{data;serialize(n,i){super.serialize(n,i);let{data:s}=this;s!==void 0&&(s.serialize(n,i),this.data=void 0)}downloadSucceeded(){let{data:n}=this;this.systemMemoryBytes=this.gpuMemoryBytes=n===void 0?0:n.numBytes,super.downloadSucceeded()}freeSystemMemory(){this.data=void 0}}return t}var El=class extends Yy(Fn){},bl=class extends Yy(fe){objectId},Gy=class extends Ne{parent=void 0;getChunk(e){let{chunks:t}=this,r=t.get(e);return r===void 0&&(r=this.getNewChunk_(wl),r.initialize(e),this.addChunk(r)),r}download(e,t){return this.parent.downloadMetadata(e,t)}};Gy=Lo([G(wy)],Gy);var cn=class extends Bn{parent;constructor(t,r){super(t,r),this.parent=t.get(r.parent)}};cn.prototype.chunkConstructor=El;var jy=class extends Ne{parent=void 0;relationshipIndex;getChunk(e){let t=on(e),{chunks:r}=this,n=r.get(t);return n===void 0&&(n=this.getNewChunk_(bl),n.initialize(t),n.objectId=e,this.addChunk(n)),n}download(e,t){return this.parent.downloadSegmentFilteredGeometry(e,this.relationshipIndex,t)}};jy=Lo([G(Ey)],jy);var Gn=class extends me{references=new Set;chunkManager;metadataChunkSource;segmentFilteredSources;constructor(t,r){super(t,r);let n=this.chunkManager=t.get(r.chunkManager),i=this.metadataChunkSource=this.registerDisposer(t.getRef(r.metadataChunkSource));this.segmentFilteredSources=r.segmentFilteredSource.map((s,o)=>{let a=this.registerDisposer(t.getRef(s));return a.parent=this,a.relationshipIndex=o,a}),i.parent=this,this.registerDisposer(n.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}recomputeChunkPriorities(){let{chunkManager:t,metadataChunkSource:r}=this;for(let n of this.references)t.requestChunk(r.getChunk(n),se.VISIBLE,GA)}add(t){throw new Error("Not implemented")}delete(t){throw new Error("Not implemented")}update(t,r){throw new Error("Not implemented")}};X(by,function(e){let t=this.get(e.id);t.references.add(e.annotation),t.chunkManager.scheduleUpdateChunkPriorities()});X(Iy,function(e){let t=this.get(e.id);t.references.delete(e.annotation),t.chunkManager.scheduleUpdateChunkPriorities()});X(_y,function(e){let t=this.get(e.id),r=e.annotationId,n=e.newAnnotation,i;r===void 0?i=t.add(n).then(s=>({...n,id:s})):n===null?i=t.delete(r).then(()=>null):i=t.update(r,n).then(()=>n),i.then(s=>{t.wasDisposed||this.invoke(ml,{id:t.rpcId,annotationId:r||n.id,newAnnotation:s})},s=>{t.wasDisposed||this.invoke(ml,{id:t.rpcId,annotationId:r||n?.id,error:s.message})})});var qy=class extends Xe(ir){localPosition;renderScaleTarget;constructor(e,t){super(e,t),this.renderScaleTarget=e.get(t.renderScaleTarget),this.localPosition=e.get(t.localPosition);let r=()=>this.chunkManager.scheduleUpdateChunkPriorities();this.registerDisposer(this.localPosition.changed.add(r)),this.registerDisposer(this.renderScaleTarget.changed.add(r)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(t),e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.state={displayDimensionRenderInfo:r.projectionParameters.value.displayDimensionRenderInfo,transformedSources:[]}}recomputeChunkPriorities(){this.chunkManager.registerLayer(this);for(let e of this.attachments.values()){let{view:t}=e,r=t.visibility.value;if(r===Number.NEGATIVE_INFINITY)continue;let n=e.state,{transformedSources:i}=n;if(i.length===0||!On(n,t.projectionParameters.value.displayDimensionRenderInfo))continue;let s=Ge(r),o=je(r),a=t.projectionParameters.value,{chunkManager:c}=this;ky(a,this.localPosition.value,this.renderScaleTarget.value,i[0],()=>{},(u,l)=>{let f=u.source.getChunk(u.curPositionInChunks);++this.numVisibleChunksNeeded,f.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable,c.requestChunk(f,s,o+0+ts*l)})}}};qy=Lo([G(Cy)],qy);X(Ay,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);n.state.transformedSources=sn(this,e.sources,r),n.state.displayDimensionRenderInfo=e.displayDimensionRenderInfo,r.chunkManager.scheduleUpdateChunkPriorities()});var Ky=class extends Et(Xe(Dr)){source;segmentationStates;constructor(e,t){super(e,t),this.source=e.get(t.source),this.segmentationStates=new jt(this.getSegmentationState(t.segmentationStates));let r=()=>this.chunkManager.scheduleUpdateChunkPriorities();this.registerDisposer(rm((n,i)=>{if(i!==void 0){for(let s of i)s!=null&&(Oo(n,s,r),Uo(n,s,r));r()}},this.segmentationStates)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>this.recomputeChunkPriorities()))}recomputeChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;let{segmentationStates:{value:t},source:{segmentFilteredSources:r}}=this;if(t===void 0)return;let{chunkManager:n}=this;n.registerLayer(this);let i=t.length;for(let s=0;s<i;++s){let o=t[s];if(o==null)continue;let a=Ge(e),c=je(e),u=r[s];xr(o,l=>{let f=u.getChunk(l);++this.numVisibleChunksNeeded,f.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable,n.requestChunk(f,a,c+jA)})}}getSegmentationState(e){if(e!==void 0)return e.map(t=>t==null?t:Sl(this.rpc,t))}};Ky=Lo([G(My)],Ky);X(Ty,function(e){let t=this.get(e.id);t.segmentationStates.value=t.getSegmentationState(e.segmentationStates)});var bt=class e extends Error{url;status;statusText;response;constructor(t,r,n,i,s){let o=`Fetching ${JSON.stringify(t)} resulted in HTTP error ${r}`;n&&(o+=`: ${n}`),o+=".",super(o,s),this.name="HttpError",this.message=o,this.url=t,this.status=r,this.statusText=n,i&&(this.response=i)}static fromResponse(t){return new e(t.url,t.status,t.statusText,t)}static fromRequestError(t,r){if(r instanceof TypeError){let n;return typeof t=="string"?n=t:n=t.url,new e(n,0,"Network or CORS error",void 0,{cause:r})}return r}},qA=32,KA=500,YA=1e4;function Il(e){return Math.min(2**e*KA,YA/2)*(1+Math.random())}async function Re(e,t){for(let r=0;;){t?.signal?.throwIfAborted();let n;try{n=await fetch(e,t)}catch(i){throw bt.fromRequestError(e,i)}if(!n.ok){let{status:i}=n;if((i===429||i===503||i===504)&&++r!==qA){await new Promise(s=>setTimeout(s,Il(r-1)));continue}throw bt.fromResponse(n)}return n}}function Jy(e){return e instanceof bt?e.status===0||e.status===403||e.status===404:!1}var JA=3;async function un(e,t,r,n,i){let s;for(let o=0;;){r.signal?.throwIfAborted(),o>1&&await new Promise(a=>setTimeout(a,Il(o-2))),s=await e.get(s,{signal:r.signal??void 0,progressListener:r.progressListener});try{return await Re(typeof t=="function"?t(s.credentials):t,n(s.credentials,r))}catch(a){if(a instanceof bt&&i(a,s.credentials)==="refresh"){if(++o===JA)throw a;continue}throw a}}}function Hy(e,t,r){return(n,i={})=>un(e,n,i,t,r)}async function Fo(e,t,r){return Re(t,r).catch(n=>{if(n.status!==500&&n.status!==401&&n.status!==403&&n.status!==504)throw n;return un(e,t,r,i=>{let s=new Headers(r.headers);return s.set("Authorization",`Bearer ${i}`),{...r,headers:s}},i=>{let{status:s}=i;if(s===403||s===401)return"refresh";throw i})})}var _l=class{baseUrl;collection;experiment;channel;resolution},Bo=class extends _l{encoding;window;static RPC_ID="boss/VolumeChunkSource";static stringify(t){return`boss:volume:${t.baseUrl}/${t.collection}/${t.experiment}/${t.channel}/${t.resolution}/${t.encoding}`}},zo=class{baseUrl;static RPC_ID="boss/MeshChunkSource";static stringify(t){return`boss:mesh:${t.baseUrl}`}};var Wy="mesh/MeshLayer",Xy="mesh/MultiscaleMeshLayer",Zy="mesh/FragmentSource",Qy="mesh/MultiscaleFragmentSource",ln=(e=>(e[e.float32=0]="float32",e[e.uint10=1]="uint10",e[e.uint16=2]="uint16",e))(ln||{});function Cl(e,t,r){return e&1|t<<1&2|r<<2&4}function t0(e,t,r,n){let i=Math.max(t,r,n),s=0,o=0,a=0,c=0;for(let u=0;u<i;++u){if(u<t){let l=Number(e>>BigInt(s++)&BigInt(1));o|=l<<u}if(u<r){let l=Number(e>>BigInt(s++)&BigInt(1));a|=l<<u}if(u<n){let l=Number(e>>BigInt(s++)&BigInt(1));c|=l<<u}}return Uint32Array.of(o,a,c)}function $o(e,t,r,n,i,s){let o=Math.max(e,t,r),a=0,c=0n;function u(l){c|=BigInt(l)<<BigInt(a++)}for(let l=0;l<o;++l)l<e&&u(n>>l&1),l<t&&u(i>>l&1),l<r&&u(s>>l&1);return c}function r0(e,t){let r=0n,n=0,i=e.length;function s(o){r|=BigInt(o&1)<<BigInt(n++)}for(let o=0;o<32;++o)for(let a=0;a<i;++a)t[a]-1>>>o&&s(e[a]>>>o);return r}function e0(e,t){return e<t&&e<(e^t)}function jn(e,t,r,n,i,s){let o=r,a=s;return e0(o^a,t^i)&&(o=t,a=i),e0(o^a,e^n)&&(o=e,a=n),o<a}function n0(e,t,r,n,i,s,o){let{octree:a,lodScales:c,chunkGridSpatialOrigin:u,chunkShape:l}=e,f=c.length-1,h=t[0],p=t[4],d=t[8],m=t[1],g=t[5],y=t[9],I=t[3],_=t[7],E=t[11],b=t[15],C=I>0?0:1,v=_>0?0:1,w=E>0?0:1,x=r[4*4],T=r[4*4+1],M=r[4*4+2],P=r[4*4+3];function F(ze,at,gt){return I*ze+_*at+E*gt+b}function S(ze,at,gt,yt,vt,dr){return F(ze+C*(yt-ze),at+v*(vt-at),gt+w*(dr-gt))}let O=F(-P*x,-P*T,-P*M),R=e.clipLowerBound[0],N=e.clipLowerBound[1],U=e.clipLowerBound[2],L=e.clipUpperBound[0],j=e.clipUpperBound[1],V=e.clipUpperBound[2],Y=Math.sqrt((h*i)**2+(m*s)**2),J=Math.sqrt((p*i)**2+(g*s)**2),de=Math.sqrt((d*i)**2+(y*s)**2),ie=Math.max(Y,J,de);function Ce(ze,at,gt){let yt=1<<ze,vt=at*5,dr=a[vt],Tr=a[vt+1],Ti=a[vt+2],Aw=a[vt+3],qh=a[vt+4],ki=dr*yt*l[0]+u[0],Ni=Tr*yt*l[1]+u[1],Di=Ti*yt*l[2]+u[2],Fs=ki+yt*l[0],Bs=Ni+yt*l[1],zs=Di+yt*l[2];if(ki=Math.max(ki,R),Ni=Math.max(Ni,N),Di=Math.max(Di,U),Fs=Math.min(Fs,L),Bs=Math.min(Bs,j),zs=Math.min(zs,V),po(ki,Ni,Di,Fs,Bs,zs,r)){let $c=Math.max(O,S(ki,Ni,Di,Fs,Bs,zs))/ie;if(gt===0||$c*n<gt){let En=c[ze];if(En!==0&&o(ze,at,En/$c,qh>>>31),ze>0&&(En===0||$c*n<En)){let Mw=En===0?gt:En,Tw=(qh&2147483647)>>>0;for(let Vc=Aw;Vc<Tw;++Vc)Ce(ze-1,Vc,Mw)}}}}Ce(f,a.length/5-1,0)}var Vo=!1;function HA(e){let t=0;for(let r=0,n=e.length;r<n;r+=3){let i=e[r],s=e[r+1],o=e[r+2],a;i>s&&(a=i,i=s,s=a),s>o&&(a=s,s=o,o=a),i>s&&(a=i,i=s,s=a),e[r]=i,e[r+1]=s,e[r+2]=o,o>t&&(t=o)}return t}var Al=0;function WA(e,t,r,n,i,s){let o=t-1>>>0,a=(i&o)>>>0;for(let c=0;;++c){let u=e[a];if(u===n)return e[a]=r,r;if(s(u))return u;++Al,a=(a+c+1&o)>>>0}}function XA(e,t){return zn(zn(0,e),t)}var ZA=1053042;function QA(e){return ZA>>>e*3&7}function Go(e){return(e>>>2)*3}function jo(e){return e&3}function Ml(e){return e>>>1}function Tl(e){return 1+(e+1>>>1)}function i0(e){return 2-e}function eM(e){return 2**Math.ceil(Math.log2(e))*4}function tM(e,t,r){let n=t.length/3,i=r.length,s=4294967295;e.fill(s),r.fill(s);for(let o=0;o<n;++o){let a=o*3;for(let c=0;c<3;++c){let u=t[a+Ml(c)],l=t[a+Tl(c)],f=o<<2|c,h=WA(r,i,f,s,XA(u,l),p=>{let d=Go(p),m=jo(p),g=t[d+Ml(m)],y=t[d+Tl(m)];return u===g&&l===y});if(h!==f){let p=Go(h),d=jo(h);e[p+d]=f,e[a+c]=h}}}return e}function rM(e,t,r,n){let i=-1>>>32-8*r.BYTES_PER_ELEMENT,o=e.length/3,a=4294967295;e:for(let c=0;c<o;++c){let u=c*3;if(e[u]!==i){for(let l=0;l<3;++l){let f=t[u+l];if(f===a)continue;let h=Go(f);if(e[h]===i)continue;let p=jo(f);r[n++]=e[u+i0(l)],r[n++]=e[u+Ml(l)],r[n++]=e[u+Tl(l)];let d=p;for(;;){if(e[u]=i,u=h,r[n++]=e[u+i0(d&3)],d=QA(d),f=t[u+(d&3)],f===a||e[h=Go(f)]===i){r[n++]=i,e[u]=i;continue e}d=jo(f)|d&4}}r[n++]=e[u],r[n++]=e[u+1],r[n++]=e[u+2],e[u]=i,r[n++]=i}}return n}function s0(e,t){if(e.length===0)return e;Al=0,t===void 0&&(t=Uint32Array.of(0,e.length));let r=0,n=0,i=0,s=0,o=0,a=HA(e),c=e.length/3*4,u=a>=65535?new Uint32Array(c):new Uint16Array(c),l=0,f=0,h=t.length-1;for(let y=0;y<h;++y)f=Math.max(f,t[y+1]-t[y]);let p=new Uint32Array(f),d=new Uint32Array(eM(f)),m=t[0];for(let y=0;y<h;++y){t[y]=l;let I=t[y+1],_=e.subarray(m,I);Vo&&(i=Date.now()),tM(p,_,d),Vo&&(s=Date.now()),l=rM(_,p,u,l),Vo&&(o=Date.now(),r+=s-i,n+=o-s),m=I}--l,t[h]=l;let g=new u.constructor(l);return g.set(u.subarray(0,l)),Vo&&console.log(`reduced from ${e.byteLength}(${e.BYTES_PER_ELEMENT}) -> ${g.byteLength}(${g.BYTES_PER_ELEMENT}): adj=${r}, emit=${n}, ${Al}/${e.length} collisions`),g}var oe=(e=>(e[e.LITTLE=0]="LITTLE",e[e.BIG=1]="BIG",e))(oe||{});function nM(){let e=Uint16Array.of(4386);return new Uint8Array(e.buffer)[0]===17?1:0}var Sr=nM();function o0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);for(let r=0,n=t.length;r<n;r+=2){let i=t[r];t[r]=t[r+1],t[r+1]=i}}function a0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);for(let r=0,n=t.length;r<n;r+=4){let i=t[r];t[r]=t[r+3],t[r+3]=i,i=t[r+1],t[r+1]=t[r+2],t[r+2]=i}}function c0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);for(let r=0,n=t.length;r<n;r+=8){let i=t[r];t[r]=t[r+7],t[r+7]=i,i=t[r+1],t[r+1]=t[r+6],t[r+6]=i,i=t[r+2],t[r+2]=t[r+5],t[r+5]=i,i=t[r+3],t[r+3]=t[r+4],t[r+4]=i}}function u0(e,t,r=Sr){t!==r&&o0(e)}function It(e,t,r=Sr){t!==r&&a0(e)}function l0(e,t,r=Sr){t!==r&&c0(e)}function qn(e,t,r,n=Sr){if(!(t===n||r===1))switch(r){case 2:o0(e);break;case 4:a0(e);break;case 8:c0(e);break}}var iM=Object.defineProperty,sM=Object.getOwnPropertyDescriptor,Ko=(e,t,r,n)=>{for(var i=n>1?void 0:n?sM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&iM(t,r,i),i};var g0=100,y0=50,oM=!1,kl=class extends fe{objectId=0n;fragmentIds;initializeManifestChunk(t,r){super.initialize(t),this.objectId=r}freeSystemMemory(){this.fragmentIds=null}serialize(t,r){super.serialize(t,r),t.fragmentIds=this.fragmentIds}downloadSucceeded(){this.systemMemoryBytes=100,this.gpuMemoryBytes=0,super.downloadSucceeded(),this.priorityTier<se.RECENT&&this.source.chunkManager.scheduleUpdateChunkPriorities()}toString(){return this.objectId.toString()}};function v0(e,t,r){let{vertexPositions:n,indices:i,vertexNormals:s,strips:o}=e;t.vertexPositions=n,t.indices=i,t.strips=o,t.vertexNormals=s;let a=n.buffer;r.push(a);let c=i.buffer;c!==a&&r.push(c),r.push(s.buffer)}function x0(e){let{vertexPositions:t,indices:r,vertexNormals:n}=e;return t.byteLength+r.byteLength+n.byteLength}var Nl=class extends fe{manifestChunk=null;fragmentId=null;meshData=null;initializeFragmentChunk(t,r,n){super.initialize(t),this.manifestChunk=r,this.fragmentId=n}freeSystemMemory(){this.manifestChunk=null,this.meshData=null,this.fragmentId=null}serialize(t,r){super.serialize(t,r),v0(this.meshData,t,r),this.meshData=null}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=x0(this.meshData),super.downloadSucceeded()}};function Yo(e,t,r){Ee(t),e.fragmentIds=re(t,r,Yt)}function Rl(e,t){let r=B.create(),n=B.create(),i=B.create(),s=new Float32Array(e.length),o=t.length;for(let c=0;c<o;c+=3){let u=t[c]*3,l=t[c+1]*3,f=t[c+2]*3;for(let h=0;h<3;++h)n[h]=e[l+h]-e[u+h],i[h]=e[f+h]-e[l+h];B.cross(r,n,i),B.normalize(r,r);for(let h=0;h<3;++h){let d=t[c+h]*3;for(let m=0;m<3;++m)s[d+m]+=r[m]}}let a=s.length;for(let c=0;c<a;c+=3){let u=s.subarray(c,c+3);B.normalize(u,u)}return s}function qo(e){return Math.min(Math.max(-127,e*127+.5),127)>>>0}function f0(e){return e<0?-1:1}function aM(e,t){let r=t.length,n=0;for(let i=0;i<r;i+=3){let s=t[i],o=t[i+1],a=t[i+2],c=1/(Math.abs(s)+Math.abs(o)+Math.abs(a));a<0?(e[n]=qo((1-Math.abs(o*c))*f0(s)),e[n+1]=qo((1-Math.abs(s*c))*f0(o))):(e[n]=qo(s*c),e[n+1]=qo(o*c)),n+=2}}function Ol(e,t,r,n,i,s,o){let a=new Float32Array(t,n,i*3);It(a,r),s===void 0&&(s=n+12*i);let c;o!==void 0&&(c=o*e);let u=c===void 0?new Uint32Array(t,s):new Uint32Array(t,s,c);if(u.length%e!==0)throw new Error(`Number of indices is not a multiple of ${e}: ${u.length}.`);return It(u,r),{vertexPositions:a,indices:u}}function Yn(e,t,r,n,i,s){return Ol(3,e,t,r,n,i,s)}var Nt=class extends Ne{fragmentSource;constructor(t,r){super(t,r);let n=this.fragmentSource=this.registerDisposer(t.getRef(r.fragmentSource));n.meshSource=this}getChunk(t){let r=on(t),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(kl),n.initializeManifestChunk(r,t),this.addChunk(n)),n}getFragmentKey(t,r){return{key:`${t}/${r}`,fragmentId:r}}getFragmentChunk(t,r){let n=this.fragmentSource,{key:i,fragmentId:s}=this.getFragmentKey(t.key,r),o=n.chunks.get(i);return o===void 0&&(o=n.getNewChunk_(Nl),o.initializeFragmentChunk(i,t,s),n.addChunk(o)),o}},h0=class extends Ne{meshSource=null;download(e,t){return this.meshSource.downloadFragment(e,t)}};h0=Ko([G(Zy)],h0);var p0=class extends an(Et(Xe(rs))){source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}attach(e){let t=()=>{this.chunkManager.scheduleUpdateChunkPriorities()},{view:r}=e;e.registerDisposer(r.visibility.changed.add(t)),e.registerDisposer(t),t()}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;this.chunkManager.registerLayer(this);let t=Ge(e),r=je(e),{source:n,chunkManager:i}=this;xr(this,s=>{let o=n.getChunk(s);++this.numVisibleChunksNeeded,i.requestChunk(o,t,r+g0);let a=o.state;if(a===z.SYSTEM_MEMORY_WORKER||a===z.SYSTEM_MEMORY||a===z.GPU_MEMORY){++this.numVisibleChunksAvailable;for(let c of o.fragmentIds){let u=n.getFragmentChunk(o,c);++this.numVisibleChunksNeeded,i.requestChunk(u,t,r+y0),u.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable}}})}};p0=Ko([G(Wy)],p0);var Dl=class extends fe{objectId=0n;manifest;initializeManifestChunk(t,r){super.initialize(t),this.objectId=r}freeSystemMemory(){this.manifest=void 0}serialize(t,r){super.serialize(t,r),t.manifest=this.manifest}downloadSucceeded(){this.systemMemoryBytes=this.manifest.octree.byteLength,this.gpuMemoryBytes=0,super.downloadSucceeded(),this.priorityTier<se.RECENT&&this.source.chunkManager.scheduleUpdateChunkPriorities()}toString(){return this.objectId.toString()}},Pl=class extends fe{subChunkOffsets=null;meshData=null;lod=0;chunkIndex=0;manifestChunk=null;freeSystemMemory(){this.meshData=this.subChunkOffsets=null}serialize(t,r){super.serialize(t,r),v0(this.meshData,t,r);let{subChunkOffsets:n}=this;t.subChunkOffsets=n,r.push(n.buffer),this.meshData=this.subChunkOffsets=null}downloadSucceeded(){let{subChunkOffsets:t}=this;this.systemMemoryBytes=this.gpuMemoryBytes=x0(this.meshData),this.systemMemoryBytes+=t.byteLength,super.downloadSucceeded()}},Kn=class extends Ne{fragmentSource;format;constructor(t,r){super(t,r);let n=this.fragmentSource=this.registerDisposer(t.getRef(r.fragmentSource));this.format=r.format,n.meshSource=this}getChunk(t){let r=on(t),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(Dl),n.initializeManifestChunk(r,t),this.addChunk(n)),n}getFragmentChunk(t,r,n){let i=`${t.key}/${r}:${n}`,s=this.fragmentSource,o=s.chunks.get(i);return o===void 0&&(o=s.getNewChunk_(Pl),o.initialize(i),o.lod=r,o.chunkIndex=n,o.manifestChunk=t,s.addChunk(o)),o}},d0=class extends Ne{meshSource=null;download(e,t){return this.meshSource.downloadFragment(e,t)}};d0=Ko([G(Qy)],d0);var cM=ge.create(),m0=class extends an(Et(Xe(rs))){source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.registerDisposer(t),t()}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;let{transform:{value:t}}=this;if(t.error!==void 0)return;let r=new Array;this.chunkManager.registerLayer(this);{let s=Ge(e),o=je(e),{source:a,chunkManager:c}=this;xr(this,u=>{let l=a.getChunk(u);++this.numVisibleChunksNeeded,c.requestChunk(l,s,o+g0);let f=l.state;(f===z.SYSTEM_MEMORY_WORKER||f===z.SYSTEM_MEMORY||f===z.GPU_MEMORY)&&(r.push(l),++this.numVisibleChunksAvailable)})}if(r.length===0)return;let{source:n,chunkManager:i}=this;for(let{view:s}of this.attachments.values()){let o=s.visibility.value;if(o===Number.NEGATIVE_INFINITY)continue;let a=Ge(o),c=je(o),u=s.projectionParameters.value,l=cM;try{Jg(l,u.displayDimensionRenderInfo,t)}catch{continue}ge.multiply(l,u.viewProjectionMat,l);let f=ho(new Float32Array(24),l),h=this.renderScaleTarget.value;for(let p of r){let d=p.manifest.lodScales.length-1;n0(p.manifest,l,f,h,u.width,u.height,(m,g,y,I)=>{if(I)return;let _=n.getFragmentChunk(p,m,g);++this.numVisibleChunksNeeded,i.requestChunk(_,a,c+y0-d+m),_.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable})}}}};m0=Ko([G(Xy)],m0);function S0(e,t){let r=Rl(e.vertexPositions,e.indices),n=new Uint8Array(r.length/3*2);aM(n,r);let i,s;oM?(i=s0(e.indices,e.subChunkOffsets),s=!0):(e.indices.BYTES_PER_ELEMENT===4&&e.vertexPositions.length/3<65535?(i=new Uint16Array(e.indices.length),i.set(e.indices)):i=e.indices,s=!1);let o;if(t===ln.uint10){let a=e.vertexPositions,c=a.length/3;o=new Uint32Array(c);for(let u=0,l=0;l<c;u+=3,++l)o[l]=a[u]&1023|(a[u+1]&1023)<<10|(a[u+2]&1023)<<20}else if(t===ln.uint16){let a=e.vertexPositions;a.BYTES_PER_ELEMENT===2?o=a:(o=new Uint16Array(a.length),o.set(a))}else o=e.vertexPositions;return{vertexPositions:o,vertexNormals:n,indices:i,strips:s}}function or(e,t,r=ln.float32){e.meshData=S0(t,r)}function Jo(e,t,r){e.meshData=S0(t,r),e.subChunkOffsets=t.subChunkOffsets}function Ho(e,t,r){let n=r;for(let i=0;i<3;++i)e[n*5+i]=e[t*5+i]>>>1;e[n*5+3]=t;for(let i=t+1;i<r;++i){let s=e[i*5]>>>1,o=e[i*5+1]>>>1,a=e[i*5+2]>>>1;(s!==e[n*5]||o!==e[n*5+1]||a!==e[n*5+2])&&(e[n*5+4]=i,++n,e[n*5]=s,e[n*5+1]=o,e[n*5+2]=a,e[n*5+3]=i)}return e[n*5+4]=r,++n,n}function w0(e,t,r,n){let i=t;for(let s=r;s<n;++s){let o=e[s*5],a=e[s*5+1],c=e[s*5+2];for(;i<r;){let u=e[i*5]>>>1,l=e[i*5+1]>>>1,f=e[i*5+2]>>>1;if(!jn(u,l,f,o,a,c))break;++i}for(e[s*5+3]=i;i<r;){let u=e[i*5]>>>1,l=e[i*5+1]>>>1,f=e[i*5+2]>>>1;if(u!==o||l!==a||f!==c)break;++i}e[s*5+4]+=i}}function De(e){return{id:e}}var E0=De("encodeCompressedSegmentationUint32"),b0=De("encodeCompressedSegmentationUint64");var Ul=0,Ll=[],Wo=new Map,Jn=new Map,uM=typeof navigator.hardwareConcurrency>"u"?4:Math.min(12,navigator.hardwareConcurrency),lM=0;function I0(e){for(let[t,r]of Wo){Wo.delete(t),r.cleanup?.(),e.postMessage(r.msg,r.transfer);return}Ll.push(e)}function fM(){++Ul;let e=new Worker(new URL("../async_computation.bundle.js",import.meta.url),{type:"module"}),t=!1;e.onmessage=r=>{if(!t){t=!0,I0(e);return}let{id:n,value:i,error:s}=r.data;I0(e);let o=Jn.get(n);Jn.delete(n),o!==void 0&&(s!==void 0?o.reject(s):o.resolve(i))}}function le(e,t,r,...n){let i=lM++,s={t:e.id,id:i,args:n};t?.throwIfAborted();let o=new Promise((c,u)=>{Jn.set(i,{resolve:c,reject:u})});if(Ll.length!==0)Ll.pop().postMessage(s,r);else{let c;if(t!==void 0){let u=function(){Wo.delete(i);let l=Jn.get(i);Jn.delete(i),l.reject(t.reason)};var a=u;t.addEventListener("abort",u,{once:!0}),c=()=>{t.removeEventListener("abort",u)}}Wo.set(i,{msg:s,transfer:r,cleanup:c}),Jn.size>Ul&&Ul<uM&&fM()}return o}async function lt(e,t,r){let{spec:n}=e.source;if(n.compressedSegmentationBlockSize!==void 0){let{dataType:i}=n,s=e.chunkDataSize,o=[s[0],s[1],s[2],s[3]||1];switch(i){case W.UINT32:e.data=await le(E0,t,[r.buffer],r,o,n.compressedSegmentationBlockSize);break;case W.UINT64:e.data=await le(b0,t,[r.buffer],r,o,n.compressedSegmentationBlockSize);break;default:throw new Error(`Unsupported data type for compressed segmentation: ${W[i]}`)}}else e.data=r}function _0(e){let t=new Uint8Array(e.buffer,e.byteOffset,e.byteLength);return t.length>=3&&t[0]===31&&t[1]===139&&t[2]===8}async function wr(e,t,r){try{let n=Fl(e instanceof Response?e:new Response(e),t,r);return await new Response(n).arrayBuffer()}catch{throw r?.throwIfAborted(),new Error(`Failed to decode ${t}`)}}function Fl(e,t,r){return e.body.pipeThrough(new DecompressionStream(t),{signal:r})}var Er=new Map;Er.set("|u1",{endianness:oe.LITTLE,dataType:W.UINT8});Er.set("|i1",{endianness:oe.LITTLE,dataType:W.INT8});for(let[e,t]of[["<",oe.LITTLE],[">",oe.BIG]]){for(let r of["u","i"])Er.set(`${e}${r}8`,{endianness:t,dataType:W.UINT64});Er.set(`${e}u2`,{endianness:t,dataType:W.UINT16}),Er.set(`${e}i2`,{endianness:t,dataType:W.INT16}),Er.set(`${e}u4`,{endianness:t,dataType:W.UINT32}),Er.set(`${e}i4`,{endianness:t,dataType:W.INT32}),Er.set(`${e}f4`,{endianness:t,dataType:W.FLOAT32})}function Bl(e){let t=Er.get(e);if(t===void 0)throw new Error(`Unsupported numpy data type: ${JSON.stringify(e)}`);return t}var zl=class{constructor(t,r,n,i){this.data=t,this.shape=r,this.dataType=n,this.fortranOrder=i}};function C0(e){if(e[0]!==147||e[1]!==78||e[2]!==85||e[3]!==77||e[4]!==80||e[5]!==89)throw new Error("Data does not match npy format.");let t=e[6],r=e[7];if(t!==1||r!==0)throw new Error(`Unsupported npy version ${t}.${r}`);let i=new DataView(e.buffer,e.byteOffset,e.byteLength).getUint16(8,!0),s=new TextDecoder("utf-8").decode(e.subarray(10,i+10)),o,a=i+10;try{o=lg(s)}catch(g){throw new Error(`Failed to parse npy header: ${g}`)}let c=o.descr,u=o.shape,l=1;if(!Array.isArray(u))throw new Error("Invalid shape ${JSON.stringify(shape)}");for(let g of u){if(typeof g!="number")throw new Error("Invalid shape ${JSON.stringify(shape)}");l*=g}let{dataType:f,endianness:h}=Bl(c),p=ut[f],d=ll[f];if(p*l+a!==e.byteLength)throw new Error("Expected length does not match length of data");let m=new d(e.buffer,e.byteOffset+a,l);return qn(m,h,p),new zl(m,u,f,o.fortran_order===!0)}async function A0(e,t,r){let n=C0(new Uint8Array(await wr(r,"deflate"))),i=e.chunkDataSize,s=e.source,{shape:o}=n;if(o.length!==3||o[0]!==i[2]||o[1]!==i[1]||o[2]!==i[0])throw new Error(`Shape ${JSON.stringify(o)} does not match chunkDataSize ${tr(i)}`);let a=n.dataType,{spec:c}=s;if(a!==c.dataType)throw new Error(`Data type ${W[a]} does not match expected data type ${W[c.dataType]}`);await lt(e,t,n.data)}var Hn=De("decodeJpeg");async function Br(e,t,r){let n=e.chunkDataSize,{uint8Array:i}=await le(Hn,t,[r],new Uint8Array(r),void 0,void 0,n[0]*n[1]*n[2],n[3]||1,!1);await lt(e,t,i)}var $l=class extends Fn{source=null;data;chunkDataSize;initializeVolumeChunk(t,r){super.initializeVolumeChunk(t,r),this.chunkDataSize=null,this.data=null}serialize(t,r){super.serialize(t,r);let n=this.chunkDataSize;n!==this.source.spec.chunkDataSize&&(t.chunkDataSize=n);let i=t.data=this.data;i!==null&&r.push(i.buffer),this.data=null}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=this.data?.byteLength??0,super.downloadSucceeded()}freeSystemMemory(){this.data=null}};function Vl(e,t){let{spec:r,tempChunkDataSize:n,tempChunkPosition:i}=e,{upperVoxelBound:s,rank:o,baseVoxelOffset:a}=r,c=r.chunkDataSize,u=n,l=zg(i,t.chunkGridPosition,c),f=!1;for(let h=0;h<o;++h){let p=Math.min(s[h],l[h]+c[h]);(u[h]=p-l[h])!==c[h]&&(f=!0)}return al(l,l,a),f?t.chunkDataSize=Uint32Array.from(u):t.chunkDataSize=c,l}var Ae=class extends Bn{tempChunkDataSize;tempChunkPosition;constructor(t,r){super(t,r);let n=this.spec.rank;this.tempChunkDataSize=new Uint32Array(n),this.tempChunkPosition=new Float32Array(n)}computeChunkBounds(t){return Vl(this,t)}};Ae.prototype.chunkConstructor=$l;var hM=Object.defineProperty,pM=Object.getOwnPropertyDescriptor,k0=(e,t,r,n)=>{for(var i=n>1?void 0:n?pM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&hM(t,r,i),i};var Gl=new Map;Gl.set("npz",A0);Gl.set("jpeg",Br);var jl=new Map;jl.set("npz","application/npygz");jl.set("jpeg","image/jpeg");function N0(e,t){return ye(Nn()(e),t)}var M0=class extends N0(Ae,Bo){chunkDecoder=Gl.get(this.parameters.encoding);async download(e,t){let{parameters:r}=this,n=`${r.baseUrl}/latest/cutout/${r.collection}/${r.experiment}/${r.channel}/${r.resolution}`;{let s=this.computeChunkBounds(e),o=e.chunkDataSize;for(let a=0;a<3;++a)n+=`/${s[a]}:${s[a]+o[a]}`}n+="/",r.window!==void 0&&(n+=`?window=${r.window[0]},${r.window[1]}`);let i=await Fo(this.credentialsProvider,n,{signal:t,headers:{Accept:jl.get(r.encoding)}});await this.chunkDecoder(e,t,await i.arrayBuffer())}};M0=k0([G()],M0);function dM(e,t){return Yo(e,t,"fragments")}function mM(e,t){let n=new DataView(t).getUint32(0,!0);or(e,Yn(t,oe.LITTLE,4,n))}var T0=class extends N0(Nt,zo){download(e,t){let{parameters:r}=this;return Fo(this.credentialsProvider,`${r.baseUrl}${e.objectId}`,{signal:t}).then(n=>n.arrayBuffer()).then(n=>dM(e,n))}downloadFragment(e,t){let{parameters:r}=this;return Fo(this.credentialsProvider,`${r.baseUrl}${e.fragmentId}`,{signal:t}).then(n=>n.arrayBuffer()).then(n=>mM(e,n))}};T0=k0([G()],T0);var Xo=new Float32Array(1);function D0(e){Xo[0]=e,e=Xo[0];for(let t=1;t<21;++t){let r=e.toPrecision(t);if(Xo[0]=parseFloat(r),Xo[0]===e)return r}return e.toString()}function P0(e){return("0"+e.toString(16)).slice(-2)}function gM(e){let t=/^rgba\(([0-9]+), ([0-9]+), ([0-9]+), (0(?:\.[0-9]+)?)\)$/;{let n=e.match(t);if(n!==null)return[parseInt(n[1],10),parseInt(n[2],10),parseInt(n[3],10),parseFloat(n[4])]}let r=/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/;{let n=e.match(r);if(n!==null)return[parseInt(n[1],16),parseInt(n[2],16),parseInt(n[3],16),1]}throw new Error(`Invalid serialized color: ${JSON.stringify(e)}.`)}function ql(e){try{if(typeof e!="string")throw new Error(`Expected string, but received ${JSON.stringify(e)}.`);let t=document.createElement("canvas").getContext("2d");t.fillStyle=e;let r=gM(t.fillStyle);return gr.fromValues(r[0]/255,r[1]/255,r[2]/255,r[3])}catch(t){throw new Error(`Failed to parse color specification: ${t.message}`)}}function R0(e){return ql(e).subarray(0,3)}function Kl(e){let t=e[3]===void 0?3:4,r=0;for(let n=0;n<t;n++)r=(r<<8>>>0)+Math.min(255,Math.max(0,Math.round(e[t-1-n]*255)));return r}function O0(e){return B.fromValues((e>>>0&255)/255,(e>>>8&255)/255,(e>>>16&255)/255)}function U0(e){return gr.fromValues((e>>>0&255)/255,(e>>>8&255)/255,(e>>>16&255)/255,(e>>>24&255)/255)}function Yl(e){if(e[3]===void 0||e[3]===1){let r="#";for(let n=0;n<3;++n)r+=P0(Math.min(255,Math.max(0,Math.round(e[n]*255))));return r}let t="rgba(";for(let r=0;r<3;++r)r!==0&&(t+=", "),t+=Math.min(255,Math.max(0,Math.round(e[r]*255)));return t+=`, ${D0(e[3])})`,t}var QB=2**-1074,yM=new Float64Array(1),ez=new Uint32Array(yM.buffer);var hz={[W.UINT8]:[0,255],[W.INT8]:[-128,127],[W.UINT16]:[0,65535],[W.INT16]:[-32768,32767],[W.UINT32]:[0,4294967295],[W.INT32]:[-2147483648,2147483647],[W.UINT64]:[0n,0xffffffffffffffffn],[W.FLOAT32]:[0,1]};var Oe=(e=>(e[e.POINT=0]="POINT",e[e.LINE=1]="LINE",e[e.AXIS_ALIGNED_BOUNDING_BOX=2]="AXIS_ALIGNED_BOUNDING_BOX",e[e.ELLIPSOID=3]="ELLIPSOID",e[e.POLYLINE=4]="POLYLINE",e))(Oe||{}),br=[0,1,2,3,4];var Iz={float32:W.FLOAT32,uint32:W.UINT32,int32:W.INT32,uint16:W.UINT16,int16:W.INT16,uint8:W.UINT8,int8:W.INT8,rgb:void 0,rgba:void 0},Hl={rgb:{serializedBytes(){return 3},alignment(){return 1},serializeCode(e,t){return`dv.setUint16(${t}, ${e}, true);dv.setUint8(${t} + 2, ${e} >>> 16);`},deserializeCode(e,t){return`${e} = dv.getUint16(${t}, true) | (dv.getUint8(${t} + 2) << 16);`},deserializeJson(e){return Kl(R0(e))},serializeJson(e){return Yl(O0(e))}},rgba:{serializedBytes(){return 4},alignment(){return 1},serializeCode(e,t){return`dv.setUint32(${t}, ${e}, true);`},deserializeCode(e,t){return`${e} = dv.getUint32(${t}, true);`},deserializeJson(e){return Kl(ql(e))},serializeJson(e){return Yl(U0(e))}},float32:{serializedBytes(){return 4},alignment(){return 4},serializeCode(e,t){return`dv.setFloat32(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getFloat32(${t}, isLittleEndian);`},deserializeJson(e){return go(e)},serializeJson(e){return e}},uint32:{serializedBytes(){return 4},alignment(){return 4},serializeCode(e,t){return`dv.setUint32(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getUint32(${t}, isLittleEndian);`},deserializeJson(e){return kr(e)},serializeJson(e){return e}},int32:{serializedBytes(){return 4},alignment(){return 4},serializeCode(e,t){return`dv.setInt32(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getInt32(${t}, isLittleEndian);`},deserializeJson(e){return kr(e)},serializeJson(e){return e}},uint16:{serializedBytes(){return 2},alignment(){return 2},serializeCode(e,t){return`dv.setUint16(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getUint16(${t}, isLittleEndian);`},deserializeJson(e){return kr(e)},serializeJson(e){return e}},int16:{serializedBytes(){return 2},alignment(){return 2},serializeCode(e,t){return`dv.setInt16(${t}, ${e}, isLittleEndian);`},deserializeCode(e,t){return`${e} = dv.getInt16(${t}, isLittleEndian);`},deserializeJson(e){return kr(e)},serializeJson(e){return e}},uint8:{serializedBytes(){return 1},alignment(){return 1},serializeCode(e,t){return`dv.setUint8(${t}, ${e});`},deserializeCode(e,t){return`${e} = dv.getUint8(${t});`},deserializeJson(e){return kr(e)},serializeJson(e){return e}},int8:{serializedBytes(){return 1},alignment(){return 1},serializeCode(e,t){return`dv.setInt8(${t}, ${e});`},deserializeCode(e,t){return`${e} = dv.getInt8(${t});`},deserializeJson(e){return kr(e)},serializeJson(e){return e}}},vM=255;function xM(e,t,r){let n=0,i=r.length,s=new Array(i),o=[];for(let h=0;h<i;++h)s[h]=h;let a=h=>Hl[r[h].type].alignment(e);s.sort((h,p)=>a(p)-a(h));let c=0,u=new Array(i),l=t,f=()=>{l+=(4-l%4)%4,n+=l,o[c]=l,l=0,++c};for(let h=0;h<i;++h){let p=s[h],d=r[p],m=Hl[d.type],g=m.serializedBytes(e),y=m.alignment(e),I=(y-l%y)%y,E=l+I+g;E+(4-E%4)%4<=vM?l+=I:f(),u[p]={offset:l,group:c},l+=g}return f(),{serializedBytes:n,offsets:u,propertyGroupBytes:o}}var os=class{constructor(t,r,n){if(this.rank=t,this.firstGroupInitialOffset=r,this.propertySpecs=n,n.length===0){this.serializedBytes=r,this.serialize=this.deserialize=()=>{},this.propertyGroupBytes=[r];return}let{serializedBytes:i,offsets:s,propertyGroupBytes:o}=xM(t,r,n);this.propertyGroupBytes=o;let a="let groupOffset0 = offset;";for(let f=1;f<o.length;++f)a+=`let groupOffset${f} = groupOffset${f-1} + ${o[f-1]}*annotationCount;`;for(let f=0;f<o.length;++f)a+=`groupOffset${f} += ${o[f]}*annotationIndex;`;let c=a,u=a,l=n.length;for(let f=0;f<l;++f){let{group:h,offset:p}=s[f],d=n[f],m=Hl[d.type],g=`properties[${f}]`,y=`groupOffset${h} + ${p}`;c+=m.serializeCode(g,y,t),u+=m.deserializeCode(g,y,t)}this.serializedBytes=i,this.serialize=new Function("dv","offset","annotationIndex","annotationCount","isLittleEndian","properties",c),this.deserialize=new Function("dv","offset","annotationIndex","annotationCount","isLittleEndian","properties",u)}serializedBytes;serialize;deserialize;propertyGroupBytes};function L0(e,t){let r=[];for(let n of br){let i=as[n];r[n]=new os(e,i.serializedBytes(e),t)}return r}function Wl(e,t,r,n,i){for(let s=0;s<n;++s)e.setFloat32(t,i[s],r),t+=4;return t}function Zo(e,t,r,n,i,s){return t=Wl(e,t,r,n,i),t=Wl(e,t,r,n,s),t}function Wn(e,t,r,n,i){for(let s=0;s<n;++s)i[s]=e.getFloat32(t,r),t+=4;return t}function Jl(e,t,r,n,i,s){return t=Wn(e,t,r,n,i),t=Wn(e,t,r,n,s),t}function SM(e,t,r,n,i,s){let o=t;for(let a=0;a<s;++a)i[a]=new Float32Array(n),o=Wn(e,o,r,n,i[a]);return o}var as={1:{icon:"\uA579",description:"Line",toJSON(e){return{pointA:Array.from(e.pointA),pointB:Array.from(e.pointB)}},restoreState(e,t,r){e.pointA=re(t,"pointA",n=>We(new Float32Array(r),n,rr)),e.pointB=re(t,"pointB",n=>We(new Float32Array(r),n,rr))},serializedBytes(e){return 2*4*e},serialize(e,t,r,n,i){Zo(e,t,r,n,i.pointA,i.pointB)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n),o=new Float32Array(n);return Jl(e,t,r,n,s,o),{type:1,pointA:s,pointB:o,id:i,properties:[]}},visitGeometry(e,t){t(e.pointA,!1),t(e.pointB,!1)},defaultProperties(e){return{properties:[],values:[]}}},4:{icon:"\u2924",description:"Polyline",toJSON(e){return{points:e.points.map(t=>Array.from(t))}},restoreState(e,t,r){e.points=re(t,"points",n=>ct(n,i=>We(new Float32Array(r),i,rr)))},serializedBytes(e){return 4*e*2+4},serialize:(e,t,r,n,i,s)=>{for(let o=0;o<i.points.length-1;o++){let a=o===i.points.length-2?1:0,c=t+o*s;e.setUint32(c,a<<31|o,r);let u=i.points[o],l=i.points[o+1];Zo(e,c+4,r,n,u,l)}},deserialize:(e,t,r,n,i,s)=>{if(s===void 0)throw new Error("Can't deserialize polyline without stride");let o=new Array;if(s==0){let a=e.getUint32(t,r)&2147483647;SM(e,t+4,r,n,o,a)}else{let a=t,c=0,u=1e5;for(;c<=u;){let l=e.getUint32(a,r)>>31,f=new Float32Array(n),h=Wn(e,a+4,r,n,f);if(o.push(f),l){let p=new Float32Array(n);Wn(e,h,r,n,p),o.push(p);break}c++,a+=s}if(c===u)throw new Error("Reached max iters on polyline deserializing")}return{type:4,points:o,id:i,properties:[]}},visitGeometry(e,t){for(let r of e.points)t(r,!1)},defaultProperties(e){return{properties:[{type:"uint32",identifier:"Num vertices",default:0,description:"Number of points in the polyline"}],values:[e.points.length]}}},0:{icon:"\u26AC",description:"Point",toJSON:e=>({point:Array.from(e.point)}),restoreState:(e,t,r)=>{e.point=re(t,"point",n=>We(new Float32Array(r),n,rr))},serializedBytes:e=>e*4,serialize:(e,t,r,n,i)=>{Wl(e,t,r,n,i.point)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n);return Wn(e,t,r,n,s),{type:0,point:s,id:i,properties:[]}},visitGeometry(e,t){t(e.point,!1)},defaultProperties(e){return{properties:[],values:[]}}},2:{icon:"\u2751",description:"Bounding Box",toJSON:e=>({pointA:Array.from(e.pointA),pointB:Array.from(e.pointB)}),restoreState:(e,t,r)=>{e.pointA=re(t,"pointA",n=>We(new Float32Array(r),n,rr)),e.pointB=re(t,"pointB",n=>We(new Float32Array(r),n,rr))},serializedBytes:e=>2*4*e,serialize(e,t,r,n,i){Zo(e,t,r,n,i.pointA,i.pointB)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n),o=new Float32Array(n);return Jl(e,t,r,n,s,o),{type:2,pointA:s,pointB:o,id:i,properties:[]}},visitGeometry(e,t){t(e.pointA,!1),t(e.pointB,!1)},defaultProperties(e){return{properties:[],values:[]}}},3:{icon:"\u25CE",description:"Ellipsoid",toJSON:e=>({center:Array.from(e.center),radii:Array.from(e.radii)}),restoreState:(e,t,r)=>{e.center=re(t,"center",n=>We(new Float32Array(r),n,rr)),e.radii=re(t,"radii",n=>We(new Float32Array(r),n,ag))},serializedBytes:e=>2*4*e,serialize(e,t,r,n,i){Zo(e,t,r,n,i.center,i.radii)},deserialize:(e,t,r,n,i)=>{let s=new Float32Array(n),o=new Float32Array(n);return Jl(e,t,r,n,s,o),{type:3,center:s,radii:o,id:i,properties:[]}},visitGeometry(e,t){t(e.center,!1),t(e.radii,!0)},defaultProperties(e){return{properties:[],values:[]}}}};function wM(e,t){let r=0,n=[],i=[];for(let f of br){let p=t[f].serializedBytes;n[f]=r;let d=e[f],m=d.length;if(f===4){i[f]=0;for(let g of d){let y=g.points.length-1;r+=p*y,i[f]+=y}}else r+=p*m}let s=[],o=[],a=[],c=new ArrayBuffer(r),u=new DataView(c),l=Sr===oe.LITTLE;for(let f of br){let h=t[f],{rank:p}=h,d=h.serialize,m=e[f];a[f]=Array.from({length:m.length},(b,C)=>C),s[f]=m.map(b=>b.id),o[f]=new Map(m.map((b,C)=>[b.id,C]));let y=as[f].serialize,I=n[f],_=h.propertyGroupBytes[0],E=0;for(let b=0,C=m.length;b<C;++b){let v=m[b];if(f===4){let w=v;y(u,I+E*_,l,p,w,_),a[f][b]=E;for(let x=0;x<w.points.length-1;x++)d(u,I,E+x,i[f],l,w.properties);E+=w.points.length-1}else y(u,I+b*_,l,p,v,_),d(u,I,b,C,l,v.properties)}f!==4&&(i[f]=m.length)}return{data:new Uint8Array(c),typeToInstanceCounts:a,typeToIds:s,typeToOffset:n,typeToIdMaps:o,typeToSize:i}}var Qo=class{constructor(t){this.propertySerializers=t}annotations=[[],[],[],[],[]];add(t){this.annotations[t.type].push(t)}serialize(){return wM(this.annotations,this.propertySerializers)}};function F0(e,t){if(!e.accessToken)return t;let r=new Headers(t.headers);return r.set("Authorization",`${e.tokenType} ${e.accessToken}`),{...t,headers:r}}function B0(e,t){let{status:r}=e;if(r===401||r===403&&!t.accessToken)return"refresh";throw e instanceof Error&&t.email!==void 0&&(e.message+=`  (Using credentials for ${JSON.stringify(t.email)})`),e}function z0(e,t,r){return e===void 0?Re(t,r):un(e,t,r,F0,B0)}function ea(e){return e===void 0?Re:Hy(e,F0,B0)}function Dt(e,t,r,n={}){return z0(t,`${e.serverUrl}${r}`,n)}var zr=(e=>(e[e.RAW=0]="RAW",e[e.JPEG=1]="JPEG",e[e.COMPRESSED_SEGMENTATION=2]="COMPRESSED_SEGMENTATION",e))(zr||{});var ta=class{instance;volumeId;scaleIndex;encoding;jpegQuality;changeSpec;static RPC_ID="brainmaps/VolumeChunkSource"},ra=class{instance;volumeId;info;changeSpec;static RPC_ID="brainmaps/MultiscaleMeshSource"},na=class{instance;volumeId;meshName;changeSpec;static RPC_ID="brainmaps/MeshSource"},ia=class{instance;volumeId;meshName;changeSpec;static RPC_ID="brainmaps/SkeletonSource"},sa=class{instance;volumeId;changestack;upperVoxelBound;static RPC_ID="brainmaps/Annotation"},oa=class{instance;volumeId;changestack;static RPC_ID="brainmaps/AnnotationSpatialIndex"};var $0="skeleton/SkeletonLayer";var EM=Object.defineProperty,bM=Object.getOwnPropertyDescriptor,IM=(e,t,r,n)=>{for(var i=n>1?void 0:n?bM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&EM(t,r,i),i};var _M=60,Xl=class extends fe{objectId=0n;vertexPositions=null;vertexAttributes=null;indices=null;initializeSkeletonChunk(t,r){super.initialize(t),this.objectId=r}freeSystemMemory(){this.vertexPositions=this.indices=null}getVertexAttributeBytes(){let t=this.vertexPositions.byteLength,{vertexAttributes:r}=this;return r?.forEach(n=>{t+=n.byteLength}),t}serialize(t,r){super.serialize(t,r);let n=this.vertexPositions,i=this.indices;t.numVertices=n.length/3,t.indices=i,r.push(i.buffer);let{vertexAttributes:s}=this;if(s!=null&&s.length>0){let o=new Uint8Array(this.getVertexAttributeBytes());o.set(new Uint8Array(n.buffer,n.byteOffset,n.byteLength));let a=t.vertexAttributeOffsets=new Uint32Array(s.length+1);a[0]=0;let c=n.byteLength;s.forEach((u,l)=>{a[l+1]=c,o.set(new Uint8Array(u.buffer,u.byteOffset,u.byteLength),c),c+=u.byteLength}),r.push(o.buffer),t.vertexAttributes=o}else t.vertexAttributes=new Uint8Array(n.buffer,n.byteOffset,n.byteLength),t.vertexAttributeOffsets=Uint32Array.of(0),n.buffer!==r[0]&&r.push(n.buffer);this.vertexPositions=this.indices=this.vertexAttributes=null}downloadSucceeded(){this.systemMemoryBytes=this.gpuMemoryBytes=this.indices.byteLength+this.getVertexAttributeBytes(),super.downloadSucceeded()}},$r=class extends Ne{getChunk(t){let r=on(t),n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(Xl),n.initializeSkeletonChunk(r,t),this.addChunk(n)),n}},V0=class extends an(Et(Xe(Dr))){source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;this.chunkManager.registerLayer(this);let t=Ge(e),r=je(e),{source:n,chunkManager:i}=this;xr(this,s=>{let o=n.getChunk(s);++this.numVisibleChunksNeeded,o.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable,i.requestChunk(o,t,r+_M)})}};V0=IM([G($0)],V0);function aa(e,t,r,n,i,s,o){let a=Ol(2,t,r,n,i,s,o);e.vertexPositions=a.vertexPositions,e.indices=a.indices}async function Xn(e,t,r){e.data=new Uint32Array(r)}async function ft(e,t,r,n=Sr,i=0,s=r.byteLength){let{spec:o}=e.source,{dataType:a}=o,c=Qi(e.chunkDataSize),u=ut[a],l=c*u;if(l!==s)throw new Error(`Raw-format chunk is ${s} bytes, but ${c} * ${u} = ${l} bytes are expected.`);let f=Co(a,r,i,s);qn(f,n,u),await lt(e,t,f)}var CM=Object.defineProperty,AM=Object.getOwnPropertyDescriptor,Zn=(e,t,r,n)=>{for(var i=n>1?void 0:n?AM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&CM(t,r,i),i};var MM=new Map([[zr.RAW,ft],[zr.JPEG,Br],[zr.COMPRESSED_SEGMENTATION,Xn]]);function Z0(e,t){e&&(t.change_spec={change_stack_id:e.changeStackId},e.timeStamp&&(t.change_spec.time_stamp=e.timeStamp),e.skipEquivalences&&(t.change_spec.skip_equivalences=e.skipEquivalences))}function Qn(e,t){return ye(Nn()(e),t)}var G0=class extends Qn(Ae,ta){chunkDecoder=MM.get(this.parameters.encoding);applyEncodingParams(e){let{encoding:t}=this.parameters;switch(t){case zr.RAW:e.subvolume_format="RAW";break;case zr.JPEG:e.subvolume_format="SINGLE_IMAGE",e.image_format_options={image_format:"JPEG",jpeg_quality:this.parameters.jpegQuality};return;case zr.COMPRESSED_SEGMENTATION:e.subvolume_format="RAW",e.image_format_options={compressed_segmentation_block_size:tr(this.spec.compressedSegmentationBlockSize)};break;default:throw new Error(`Invalid encoding: ${t}`)}}async download(e,t){let{parameters:r}=this,n=this.computeChunkBounds(e),i=e.chunkDataSize,s=`/v1/volumes/${r.volumeId}/subvolume:binary`,o={geometry:{corner:tr(n),size:tr(i),scale:r.scaleIndex}};this.applyEncodingParams(o),Z0(r.changeSpec,o);let a=await Dt(r.instance,this.credentialsProvider,s,{method:"POST",body:JSON.stringify(o),signal:t});await this.chunkDecoder(e,t,await a.arrayBuffer())}};G0=Zn([G()],G0);function TM(e,t,r,n){let i=yr(BigInt("0x"+e));return t0(i,t,r,n)}function kM(e,t){Ee(t);let r=e.source,n=re(t,"fragmentKey",Yt),i=re(t,"supervoxelId",Yt),s=n.length;if(s!==i.length)throw new Error("Expected fragmentKey and supervoxelId arrays to have the same length.");let o=new Map;n.forEach((E,b)=>{let C=o.get(E);C===void 0&&(C=[],o.set(E,C)),C.push(i[b])});let{chunkShape:a}=r.parameters.info,c=r.parameters.info.lods[0].gridShape,u=Math.ceil(Math.log2(c[0])),l=Math.ceil(Math.log2(c[1])),f=Math.ceil(Math.log2(c[2])),h=Array.from(o.entries()).map(([E,b])=>({fragmentId:E,corner:TM(E,u,l,f),supervoxelIds:b}));h.sort((E,b)=>jn(E.corner[0],E.corner[1],E.corner[2],b.corner[0],b.corner[1],b.corner[2])?-1:1);let p,d,m=0,g;if(s===0)p=d=qi,g=Uint32Array.of(0,0,0,0,2147483648);else{let E=B.clone(tg),b=B.clone(qi);for(h.forEach(C=>{let{corner:v}=C;for(let w=0;w<3;++w)E[w]=Math.min(E[w],v[w]),b[w]=Math.max(b[w],v[w])}),m=1;b[0]>>>m-1!==E[0]>>>m-1||b[1]>>>m-1!==E[1]>>>m-1||b[2]>>>m-1!==E[2]>>>m-1;)++m;p=B.multiply(E,E,a),d=B.add(b,B.multiply(b,b,a),a)}let{lods:y}=r.parameters.info,I=new Float32Array(Math.max(y.length,m));for(let E=0;E<y.length;++E)I[E]=y[E].scale;if(s!==0){let E=new Uint32Array(h.length*I.length*5);h.forEach((v,w)=>{E.set(v.corner,w*5),E[w*5]=v.corner[0]});let b=0,C=h.length;for(let v=1;v<I.length;++v){let w=Ho(E,b,C);b=C,C=w}g=E.slice(0,C*5)}let _={chunkShape:a,chunkGridSpatialOrigin:qi,clipLowerBound:p,clipUpperBound:d,octree:g,lodScales:I,vertexOffsets:new Float32Array(I.length*3)};e.manifest=_,e.fragmentSupervoxelIds=h}var Ql=255;function Q0(e,t){let r=e.byteLength,n=0,i=new DataView(e),s=32;for(;n<r;){if(n+s>r)throw new Error("Invalid batch mesh fragment response.");let c=i.getBigUint64(n,!0).toString()+"\0";n+=8;let u=i.getUint32(n,!0),l=i.getUint32(n+4,!0);if(n+=8,l!==0)throw new Error("Invalid batch mesh fragment response.");if(n+u+8+8>r)throw new Error("Invalid batch mesh fragment response.");let f=new TextDecoder().decode(new Uint8Array(e,n,u)),h=c+f;n+=u;let p=i.getUint32(n,!0),d=i.getUint32(n+4,!0);n+=8;let m=i.getUint32(n,!0),g=i.getUint32(n+4,!0);if(n+=8,d!==0||g!==0)throw new Error("Invalid batch mesh fragment response.");let y=n+m*12+p*12;if(y>r)throw new Error("Invalid batch mesh fragment response.");t({fullKey:h,buffer:e,verticesOffset:n,numVertices:p,indicesOffset:n+12*p,numIndices:m*3}),n=y}}function ev(e){let t=0,r=0;for(let a of e)t+=a.numVertices,r+=a.numIndices;let n=new Float32Array(t*3),i=new Uint32Array(r),s=0,o=0;for(let a of e){n.set(new Float32Array(a.buffer,a.verticesOffset,a.numVertices*3),s*3);let{numIndices:c}=a,u=new Uint32Array(a.buffer,a.indicesOffset,c);It(u,oe.LITTLE);for(let l=0;l<c;++l)i[o++]=u[l]+s;s+=a.numVertices}return It(n,oe.LITTLE),{vertexPositions:n,indices:i}}async function tv(e,t,r,n){let i="/v1/objects/meshes:batch",s=[],o,a=0,c=new Map;for(let[l,f]of r){c.set(l,f),r.delete(l);let h=l.indexOf("\0"),p=l.substring(0,h),d=l.substring(h+1);if(p!==o&&s.push({object_id:p,fragment_keys:[]}),s[s.length-1].fragment_keys.push(d),++a===Ql)break}let u={volume_id:t.volumeId,mesh_name:t.meshName,batches:s};try{return await(await Dt(t.instance,e,i,{method:"POST",body:JSON.stringify(u),signal:n})).arrayBuffer()}finally{for(let[l,f]of c)r.set(l,f)}}var j0=class extends Qn(Kn,ra){listFragmentsParams=(()=>{let{parameters:e}=this,{changeSpec:t}=e;return t!==void 0?`&header.changeStackId=${t.changeStackId}`:""})();download(e,t){let{parameters:r}=this,n=`/v1/objects/${r.volumeId}/meshes/${r.info.lods[0].info.name}:listfragments?object_id=${e.objectId}&return_supervoxel_ids=true`+this.listFragmentsParams;return Dt(r.instance,this.credentialsProvider,n,{signal:t}).then(i=>i.json()).then(i=>kM(e,i))}async downloadFragment(e,t){let{parameters:r}=this,n=e.manifestChunk,{fragmentSupervoxelIds:i}=n,s=n.manifest,{lod:o}=e,{octree:a}=s,c=i.length,u=e.chunkIndex,l=u;for(;l>=c;)l=a[l*5+3];let f=u+1;for(;f>c;)f=a[f*5-1]&2147483647;let{relativeBlockShape:h,gridShape:p}=r.info.lods[o],d=Math.ceil(Math.log2(p[0])),m=Math.ceil(Math.log2(p[1])),g=Math.ceil(Math.log2(p[2])),y=new Map;for(let M=l;M<f;++M){let P=Math.floor(a[M*5]/h[0]),F=Math.floor(a[M*5+1]/h[1]),S=Math.floor(a[M*5+2]/h[2]),O=$o(d,m,g,P,F,S).toString(16).padStart(16,"0"),R=i[M];for(let N of R.supervoxelIds)y.set(N+"\0"+O,M)}let I=Math.max(0,o-1),_=[],E=Array.from(y);E.sort((M,P)=>St(M[0],P[0])),y=new Map(E);let b=r.info.lods[o].info.name,C=!0;await new Promise((M,P)=>{let F=0,S=!1,O=()=>{if(!S){for(;y.size!==0&&(++F,tv(this.credentialsProvider,{instance:r.instance,volumeId:r.volumeId,meshName:b},y,t).then(R=>{--F,Q0(R,N=>{let U=y.get(N.fullKey);if(!y.delete(N.fullKey))throw new Error(`Received unexpected fragment key: ${JSON.stringify(N.fullKey)}.`);N.chunkIndex=U,_.push(N)}),O()}).catch(R=>{S=!0,P(R)}),!!C););if(e.downloadSlots=Math.max(1,F),F===0){M(void 0);return}}};O()}),_.sort((M,P)=>M.chunkIndex-P.chunkIndex);let v=0,w=1<<3*(o-I),x=new Uint32Array(w+1),T=0;for(let M of _){let P=M.chunkIndex,F=Cl(a[P*5]>>>I,a[P*5+1]>>>I,a[P*5+2]>>>I)&w-1;x.fill(v,T+1,F+1),T=F,v+=M.numIndices}x.fill(v,T+1,w+1),Jo(e,{...ev(_),subChunkOffsets:x},ln.float32)}};j0=Zn([G()],j0);function NM(e){let t=[],r=0,n=e.length;for(;r<n;)t.push(JSON.stringify(e.slice(r,r+Ql))),r+=Ql;return t}function DM(e,t){Ee(t);let r=re(t,"fragmentKey",Yt),n=re(t,"supervoxelId",Yt);if(r.length!==n.length)throw new Error("Expected fragmentKey and supervoxelId arrays to have the same length.");let s=n.map((o,a)=>o+"\0"+r[a]);e.fragmentIds=NM(s)}var q0=class extends Qn(Nt,na){listFragmentsParams=(()=>{let{parameters:e}=this,{changeSpec:t}=e;return t!==void 0?`&header.changeStackId=${t.changeStackId}`:""})();download(e,t){let{parameters:r}=this,n=`/v1/objects/${r.volumeId}/meshes/${r.meshName}:listfragments?object_id=${e.objectId}&return_supervoxel_ids=true`+this.listFragmentsParams;return Dt(r.instance,this.credentialsProvider,n,{signal:t}).then(i=>i.json()).then(i=>DM(e,i))}async downloadFragment(e,t){let{parameters:r}=this,n=new Map;for(let o of JSON.parse(e.fragmentId))n.set(o,null);let i=[],{credentialsProvider:s}=this;for(;n.size!==0;){let o=await tv(s,r,n,t);Q0(o,a=>{if(!n.delete(a.fullKey))throw new Error(`Received unexpected fragment key: ${JSON.stringify(a.fullKey)}.`);i.push(a)})}or(e,ev(i))}};q0=Zn([G()],q0);function PM(e,t){let r=new DataView(t),n=r.getUint32(0,!0);if(r.getUint32(4,!0)!==0)throw new Error("The number of vertices should not exceed 2^32-1.");let s=r.getUint32(8,!0);if(r.getUint32(12,!0)!==0)throw new Error("The number of edges should not exceed 2^32-1.");aa(e,t,oe.LITTLE,16,n,void 0,s)}var K0=class extends Qn($r,ia){download(e,t){let{parameters:r}=this,n={object_id:`${e.objectId}`},i=`/v1/objects/${r.volumeId}/meshes/${r.meshName}/skeleton:binary`;return Z0(r.changeSpec,n),Dt(r.instance,this.credentialsProvider,i,{method:"POST",body:JSON.stringify(n),signal:t}).then(s=>s.arrayBuffer()).then(s=>PM(e,s))}};K0=Zn([G()],K0);var tf=["LOCATION","LINE","VOLUME"];function Y0(e){let t=/(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)/,r=e.match(t);if(r===null)throw new Error(`Error parsing number triplet: ${JSON.stringify(e)}.`);return B.fromValues(parseFloat(r[1]),parseFloat(r[2]),parseFloat(r[3]))}function ef(e){return e.volumeId+":"+e.changestack+":"}function rv(e,t){if(!t.startsWith(e))throw new Error(`Received annotation id ${JSON.stringify(t)} does not have expected prefix of ${JSON.stringify(e)}.`);return t.substring(e.length)}function RM(e){if(e!=null)return[BigUint64Array.from(ct(e,yr))]}function nv(e,t,r){let n=re(e,"corner",l=>Y0(et(l))),i=re(e,"size",l=>Y0(et(l))),s=re(e,"payload",Ju),o=re(e,"type",et),a=re(e,"id",et),c=rv(t,a),u=re(e,"objectLabels",RM);if(r!==void 0&&c!==r)throw new Error(`Received annotation has unexpected id ${JSON.stringify(a)}.`);switch(o){case"LOCATION":{if(B.equals(i,qi))return{type:Oe.POINT,id:c,point:n,description:s,relatedSegments:u,properties:[]};let l=B.scale(B.create(),i,.5),f=B.add(B.create(),n,l);return{type:Oe.ELLIPSOID,id:c,center:f,radii:l,description:s,relatedSegments:u,properties:[]}}case"LINE":return{type:Oe.LINE,id:c,pointA:n,pointB:B.add(B.create(),n,i),description:s,relatedSegments:u,properties:[]};case"VOLUME":return{type:Oe.AXIS_ALIGNED_BOUNDING_BOX,id:c,pointA:n,pointB:B.add(B.create(),n,i),description:s,relatedSegments:u,properties:[]};default:throw new Error(`Unknown spatial annotation type: ${JSON.stringify(o)}.`)}}function OM(e,t,r){Ee(e);let n=re(e,"annotations",i=>We([void 0],i,Ee))[0];return nv(n,t,r)}var UM=L0(3,[]);function iv(e,t){let r=new Qo(UM),n=e.source.parent,i=ef(n.parameters);t.forEach((s,o)=>{try{Ee(s);let a=re(s,"annotations",c=>c===void 0?[]:c);if(!Array.isArray(a))throw new Error(`Expected array, but received ${JSON.stringify(typeof a)}.`);for(let c of a)try{r.add(nv(c,i))}catch(u){throw new Error(`Error parsing annotation: ${u.message}`)}}catch(a){throw new Error(`Error parsing ${tf[o]} annotations: ${a.message}`)}}),e.data=Object.assign(new Vn,r.serialize())}function J0(e){let t=e.indexOf(".");return e.substring(0,t)}function fn(e){return`${Math.round(e[0])},${Math.round(e[1])},${Math.round(e[2])}`}function Zl(e,t){return`${e.volumeId}:${e.changestack}:${t}`}function H0(e){let t=e.description||"",r=e.relatedSegments===void 0?void 0:Array.from(e.relatedSegments[0],n=>n.toString());switch(e.type){case Oe.LINE:{let{pointA:n,pointB:i}=e,s=B.subtract(B.create(),i,n);return{type:"LINE",corner:fn(n),size:fn(s),object_labels:r,payload:t}}case Oe.AXIS_ALIGNED_BOUNDING_BOX:{let{pointA:n,pointB:i}=e,s=cl(B.create(),n,i),o=$g(B.create(),n,i),a=B.subtract(o,o,s);return{type:"VOLUME",corner:fn(s),size:fn(a),object_labels:r,payload:t}}case Oe.POINT:return{type:"LOCATION",corner:fn(e.point),size:"0,0,0",object_labels:r,payload:t};case Oe.ELLIPSOID:{let n=B.subtract(B.create(),e.center,e.radii),i=B.scale(B.create(),e.radii,2);return{type:"LOCATION",corner:fn(n),size:fn(i),object_labels:r,payload:t}}}}var W0=class extends Qn(cn,oa){async download(e,t){let{parameters:r}=this;return Promise.all(tf.map(n=>Dt(r.instance,this.credentialsProvider,`/v1/changes/${r.volumeId}/${r.changestack}/spatials:get`,{signal:t,method:"POST",body:JSON.stringify({type:n,ignore_payload:!0})}).then(i=>i.json()))).then(n=>{iv(e,n)})}};W0=Zn([G()],W0);var X0=class extends Qn(Gn,sa){downloadSegmentFilteredGeometry(e,t,r){let{parameters:n}=this;return Promise.all(tf.map(i=>Dt(n.instance,this.credentialsProvider,`/v1/changes/${n.volumeId}/${n.changestack}/spatials:get`,{signal:r,method:"POST",body:JSON.stringify({type:i,object_labels:[e.objectId.toString()],ignore_payload:!0})}).then(s=>s.json()))).then(i=>{iv(e,i)})}downloadMetadata(e,t){let{parameters:r}=this,n=e.key;return Dt(r.instance,this.credentialsProvider,`/v1/changes/${r.volumeId}/${r.changestack}/spatials:get`,{signal:t,method:"POST",body:JSON.stringify({type:J0(n),id:Zl(r,n)})}).then(i=>i.json()).then(i=>{e.annotation=OM(i,ef(r),n)},()=>{e.annotation=null})}add(e){let{parameters:t}=this,r=H0(e);return Dt(t.instance,this.credentialsProvider,`/v1/changes/${t.volumeId}/${t.changestack}/spatials:push`,{method:"POST",body:JSON.stringify({annotations:[r]})}).then(n=>n.json()).then(n=>{Ee(n);let i=re(n,"ids",Yt);if(i.length!==1)throw new Error(`Expected list of 1 id, but received ${JSON.stringify(i)}.`);let s=ef(this.parameters);return rv(s,i[0])})}update(e,t){let{parameters:r}=this,n=H0(t);return n.id=Zl(r,e),Dt(r.instance,this.credentialsProvider,`/v1/changes/${r.volumeId}/${r.changestack}/spatials:push`,{method:"POST",body:JSON.stringify({annotations:[n]})}).then(i=>i.json())}delete(e){let{parameters:t}=this;return Dt(t.instance,this.credentialsProvider,`/v1/changes/${t.volumeId}/${t.changestack}/spatials:delete`,{method:"POST",body:JSON.stringify({type:J0(e),ids:[Zl(t,e)]})}).then(r=>r.json())}};X0=Zn([G()],X0);var hn=De("decodePng");var cs=(e=>(e[e.JPG=0]="JPG",e[e.JPEG=1]="JPEG",e[e.PNG=2]="PNG",e))(cs||{}),ca=class{url;encoding;format;tilesize;overlap;static RPC_ID="deepzoom/ImageTileSource"};var LM=Object.defineProperty,FM=Object.getOwnPropertyDescriptor,BM=(e,t,r,n)=>{for(var i=n>1?void 0:n?FM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&LM(t,r,i),i};var sv=class extends ye(ke(Ae),ca){tileKvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);gridShape=(()=>{let e=new Uint32Array(2),{upperVoxelBound:t,chunkDataSize:r}=this.spec;for(let n=0;n<2;++n)e[n]=Math.ceil(t[n]/r[n]);return e})();async download(e,t){let{parameters:r}=this,{tilesize:n,overlap:i,encoding:s}=r,[o,a]=e.chunkGridPosition,c=o===0?0:i,u=a===0?0:i,l=`${this.tileKvStore.path}/${o}_${a}.${r.format}`,f=await this.tileKvStore.store.read(l,{signal:t});if(f===void 0)return;let h=new Uint8Array(await f.response.arrayBuffer()),p=0,d=0,m;switch(s){case cs.PNG:{let g=await le(hn,t,[h.buffer],h,void 0,void 0,void 0,3,1,!1);({width:p,height:d}=g),m=Qm(g.uint8Array,p*d,3);break}case cs.JPG:case cs.JPEG:{({uint8Array:m,width:p,height:d}=await le(Hn,t,[h.buffer],h,void 0,void 0,void 0,3,!1));break}}if(m!==void 0){let g=n*n,y=p*d,I=e.data=new Uint8Array(g*3);for(let _=0;_<3;_++)for(let E=0;E<d;E++)for(let b=0;b<p;b++)I[b+E*n+_*g]=m[b+c+(E+u)*p+_*y]}}};sv=BM([G()],sv);var ua=class{constructor(t,r){this.baseUrl=t,this.nodeKey=r}getNodeApiUrl(t=""){return`${this.baseUrl}/api/node/${this.nodeKey}${t}`}getRepoInfoUrl(){return`${this.baseUrl}/api/repos/info`}getKeyValueUrl(t,r){return`${this.getNodeApiUrl()}/${t}/key/${r}`}getKeyValueRangeUrl(t,r,n){return`${this.getNodeApiUrl()}/${t}/keyrange/${r}/${n}`}getKeyValuesUrl(t){return`${this.getNodeApiUrl()}/${t}/keyvalues?jsontar=false`}};function la(e,t){return e.includes("?")?e+="&":e+="?",e+="app=Neuroglancer",t&&(e+=`&u=${t}`),e}function fa(e,t,r){return un(e,t,r,(n,i)=>{let s={...i};return n.token&&(s.headers={...s.headers,Authorization:`Bearer ${n}`}),s},n=>{let{status:i}=n;if(i===403||i===401)return"refresh";throw n})}var Vr=(e=>(e[e.JPEG=0]="JPEG",e[e.RAW=1]="RAW",e[e.COMPRESSED_SEGMENTATION=2]="COMPRESSED_SEGMENTATION",e[e.COMPRESSED_SEGMENTATIONARRAY=3]="COMPRESSED_SEGMENTATIONARRAY",e))(Vr||{}),us=class{baseUrl;nodeKey;dataInstanceKey;authServer;user},ha=class extends us{dataScale;encoding;static RPC_ID="dvid/VolumeChunkSource"},pa=class extends us{static RPC_ID="dvid/SkeletonSource"},da=class extends us{static RPC_ID="dvid/MeshSource"};function ov(e,t){let r=zM(t);if(r.length<1)throw new Error("ERROR parsing swc data");let n=new Uint32Array(r.length),i=0,s=0;r.forEach((l,f)=>{l&&(n[f]=i++,l.parent>=0&&++s)});let o=new Float32Array(3*i),a=new Uint32Array(2*s),c=0,u=0;r.forEach(l=>{l&&(o[3*c]=l.x,o[3*c+1]=l.y,o[3*c+2]=l.z,l.parent>=0&&(a[2*u]=c,a[2*u+1]=n[l.parent],++u),++c)}),e.indices=a,e.vertexPositions=o}function zM(e){let t=e.split(`
`),r=[],n="-?\\d*(?:\\.\\d+)?",i=new RegExp("^[ \\t]*("+["\\d+","\\d+",n,n,n,n,"-1|\\d+"].join(")[ \\t]+(")+")[ \\t]*$");return t.forEach(s=>{let o=s.match(i);if(o){let a=r[parseInt(o[1],10)]=new rf;a.type=parseInt(o[2],10),a.x=parseFloat(o[3]),a.y=parseFloat(o[4]),a.z=parseFloat(o[5]),a.radius=parseFloat(o[6]),a.parent=parseInt(o[7],10)}}),r}var rf=class{type;x;y;z;radius;parent};var $M=Object.defineProperty,VM=Object.getOwnPropertyDescriptor,nf=(e,t,r,n)=>{for(var i=n>1?void 0:n?VM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&$M(t,r,i),i};function sf(e,t){return ye(Nn()(e),t)}var av=class extends sf($r,pa){download(e,t){let{parameters:r}=this,n=`${e.objectId}`,i=`${r.baseUrl}/api/node/${r.nodeKey}/${r.dataInstanceKey}/key/`+n+"_swc";return fa(this.credentialsProvider,la(i,r.user),{signal:t}).then(s=>s.arrayBuffer()).then(s=>{let o=new TextDecoder("utf-8");ov(e,o.decode(s))})}};av=nf([G()],av);function GM(e,t){let n=new DataView(t).getUint32(0,!0);or(e,Yn(t,oe.LITTLE,4,n))}var cv=class extends sf(Nt,da){download(e){return e.fragmentIds=[`${e.objectId}`],Promise.resolve(void 0)}downloadFragment(e,t){let{parameters:r}=this,i=new ua(r.baseUrl,r.nodeKey).getKeyValueUrl(r.dataInstanceKey,`${e.fragmentId}.ngmesh`);return fa(this.credentialsProvider,la(i,r.user),{signal:t}).then(s=>s.arrayBuffer()).then(s=>GM(e,s))}};cv=nf([G()],cv);var uv=class extends sf(Ae,ha){async download(e,t){let r=this.parameters,n;{let o=this.computeChunkBounds(e),a=e.chunkDataSize;n=this.getPath(o,a)}let i=this.getDecoder(r),s=await fa(this.credentialsProvider,la(`${r.baseUrl}${n}`,r.user),{signal:t}).then(o=>o.arrayBuffer());await i(e,t,r.encoding===Vr.JPEG?s.slice(16):s)}getPath(e,t){let r=this.parameters;return r.encoding===Vr.JPEG?`/api/node/${r.nodeKey}/${r.dataInstanceKey}/subvolblocks/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}`:r.encoding===Vr.RAW?`/api/node/${r.nodeKey}/${r.dataInstanceKey}/raw/0_1_2/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}/jpeg`:r.encoding===Vr.COMPRESSED_SEGMENTATIONARRAY?`/api/node/${r.nodeKey}/${r.dataInstanceKey}/raw/0_1_2/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}?compression=googlegzip&scale=${r.dataScale}`:`/api/node/${r.nodeKey}/${r.dataInstanceKey}/raw/0_1_2/${t[0]}_${t[1]}_${t[2]}/${e[0]}_${e[1]}_${e[2]}?compression=googlegzip`}getDecoder(e){return e.encoding===Vr.JPEG||e.encoding===Vr.RAW?Br:Xn}};uv=nf([G()],uv);function ma(e,t){if(t===void 0)return{outer:e,inner:{offset:0,length:e.length}};if("suffixLength"in t){let r=Math.min(e.length,t.suffixLength);return{outer:{offset:e.offset+(e.length-r),length:r},inner:{offset:e.length-r,length:r}}}if(t.offset+t.length>e.length)throw new Error(`Requested byte range ${JSON.stringify(t)} not valid for value of length ${e.length}`);return{outer:{offset:e.offset+t.offset,length:t.length},inner:t}}function ls(e,t){let{outer:{offset:r,length:n}}=ma({offset:0,length:e.length},t);return{offset:r,length:n,totalSize:e.length,response:new Response(e.subarray(r,r+n))}}var tt=class{constructor(t,r){this.base=t,this.byteRange=r}async stat(t){return{totalSize:this.byteRange.length}}async read(t){let{byteRange:r}=this,{outer:n,inner:i}=ma(r,t.byteRange);return n.length===0?{response:new Response(new Uint8Array(0)),totalSize:r.length,...i}:{response:(await Pr(this.base,{signal:t.signal,byteRange:n,strictByteRange:!0,throwIfMissing:!0})).response,totalSize:r.length,...i}}getUrl(){let{offset:t,length:r}=this.byteRange;return`${this.base.getUrl()}|range:${t}-${t+r}`}};function of(e){if(e!==void 0)return`bytes=${e.offset}-${e.offset+e.length-1}`}var jM=navigator.userAgent.indexOf("Chrome")!==-1?"no-store":"default";function af(e,t){return new URL(e).pathname+"/"===new URL(t.url).pathname}function lv(e){let t=e.match(/bytes ([0-9]+)-([0-9]+)\/([0-9]+|\*)/);if(t===null)throw new Error(`Invalid content-range header: ${JSON.stringify(e)}`);let r=parseInt(t[1],10),n=parseInt(t[2],10),i;t[3]!=="*"&&(i=parseInt(t[3],10));let s=n-r+1;return{offset:r,length:s,totalSize:i}}async function ei(e,t,r,n,i=Re){let s;try{let{byteRange:o}=n,a;if(o!==void 0)if("suffixLength"in o){let p=await pn(e,t,r,n,i);if(p===void 0)return;let{totalSize:d}=p;if(d===void 0)throw new Error(`Failed to determine total size of ${e.getUrl(t)} in order to fetch suffix ${JSON.stringify(o)}`);if(s=ma({offset:0,length:d},o).outer,s.length===0)return{...s,totalSize:d,response:new Response(new Uint8Array(0))};a=of(s)}else s=o,s.length===0?a=of({offset:Math.max(s.offset-1,0),length:1}):a=of(s);let c={signal:n.signal,progressListener:n.progressListener};a!==void 0&&(c.headers={range:a},c.cache=jM);let u=await i(r,c);if(af(r,u))return;let l,f,h;if(u.status===206){let p=u.headers.get("content-range");if(p===null)if(s!==void 0)l=s.offset;else throw new Error("Unexpected HTTP 206 response when no byte range specified.");p!==null&&({offset:l,length:f,totalSize:h}=lv(p))}else f=h=ga(u.headers);return l===void 0&&(l=0),f===void 0&&(f=ga(u.headers)),s?.length===0&&(u=new Response(new Uint8Array(0)),l=s.offset,f=0),{response:u,offset:l,length:f,totalSize:h}}catch(o){return o instanceof bt&&o.status===416&&s?.length===0&&s.offset===0?{response:new Response(new Uint8Array(0)),offset:0,length:0,totalSize:0}:cf(e,t,n,o)}}function ga(e){let t=e.get("content-length");if(e.get("content-encoding")===null&&t!==null){let n=Number(t);if(!Number.isFinite(n)||n<0)throw new Error("Invalid content-length: {contentLength}");return n}}function cf(e,t,r,n){if(Jy(n)){if(r.throwIfMissing===!0)throw new Wi(new Te(e,t),{cause:n});return}throw n}async function pn(e,t,r,n,i=Re){try{let s=await i(r,{method:"HEAD",signal:n.signal,progressListener:n.progressListener});return af(r,s)?void 0:{totalSize:ga(s.headers)}}catch(s){if(!(s instanceof bt&&(s.status===405||s.status===501)))return cf(e,t,n,s)}try{let s=await i(r,{signal:n.signal,progressListener:n.progressListener,headers:{range:"bytes=0-0"}});if(af(r,s))return;let o;if(s.status===200)o=ga(s.headers);else{let a=s.headers.get("content-range");a!==null&&({totalSize:o}=lv(a))}return{totalSize:o}}catch(s){return s instanceof bt&&s.status===416?{totalSize:0}:cf(e,t,n,s)}}var ti=class{constructor(t,r,n=r,i=Re){this.sharedKvStoreContext=t,this.baseUrl=r,this.baseUrlForDisplay=n,this.fetchOkImpl=i}stat(t,r){return pn(this,t,Ht(this.baseUrl,t),r,this.fetchOkImpl)}read(t,r){return ei(this,t,Ht(this.baseUrl,t),r,this.fetchOkImpl)}getUrl(t){return Ht(this.baseUrlForDisplay,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function qM(e,t,r){return{scheme:e,description:`${e} (unauthenticated)`,getKvStore(n){try{let{baseUrl:i,path:s}=Dn(n.url);return{store:new r(t,i),path:s}}catch(i){throw new Error(`Invalid URL ${JSON.stringify(n.url)}`,{cause:i})}}}}function fv(e,t){for(let r of["http","https"])e.registerBaseKvStoreProvider(n=>qM(r,n,t))}var hv="GrapheneMeshSource:NewSegment";var ya=class{url;static RPC_ID="graphene/ChunkedGraphSource"},va=class{manifestUrl;fragmentUrl;lod;sharding;nBitsForLayerId;static RPC_ID="graphene/MeshSource"};function uf(e,t){return e>>BigInt(64-t)==1n}function pv(e){if(e.charAt(0)==="~"){let r=e.substring(1).split(/:(.+)/);return{key:r[0],fragmentId:r[1]}}return{key:e,fragmentId:e}}var dv="ChunkedGraphLayer",mv="ChunkedGraphLayer:updateSources",gv=5;async function yv(e){if(e.response){let t;return e.response.headers.get("content-type")==="application/json"?t=(await e.response.json()).message:t=await e.response.text(),t}}function lf(e,t){let{store:r,path:n}=e.getKvStore(t);if(!(r instanceof ti))throw new Error(`Non-HTTP URL ${JSON.stringify(t)} not supported`);let{fetchOkImpl:i,baseUrl:s}=r;if(s.includes("?"))throw new Error(`Invalid URL ${s}: query parameters not supported`);return{fetchOkImpl:i,baseUrl:Ht(s,n)}}var Gr=(e=>(e[e.RAW=0]="RAW",e[e.JPEG=1]="JPEG",e[e.COMPRESSED_SEGMENTATION=2]="COMPRESSED_SEGMENTATION",e[e.COMPRESSO=3]="COMPRESSO",e[e.PNG=4]="PNG",e[e.JXL=5]="JXL",e))(Gr||{}),xa=class{url;encoding;sharding;static RPC_ID="precomputed/VolumeChunkSource"},Sa=class{url;lod;static RPC_ID="precomputed/MeshSource"},ri=(e=>(e[e.RAW=0]="RAW",e[e.GZIP=1]="GZIP",e))(ri||{}),_a=(e=>(e[e.IDENTITY=0]="IDENTITY",e[e.MURMURHASH3_X86_128=1]="MURMURHASH3_X86_128",e))(_a||{});var wa=class{url;metadata;static RPC_ID="precomputed/MultiscaleMeshSource"},Ea=class{url;metadata;static RPC_ID="precomputed/SkeletonSource"},ba=class{url;sharding;static RPC_ID="precomputed/AnnotationSpatialIndexSource"},Ia=class{rank;relationships;properties;byId;type;static RPC_ID="precomputed/AnnotationSource"};var vv=Symbol("objectId"),YM=0;function xv(e){if(e instanceof Object){let t=e[vv];return t===void 0&&(t=e[vv]=YM++),`o${t}`}return""+JSON.stringify(e)}var ff=class extends fe{asyncMemoize;initialize(t){super.initialize(t)}freeSystemMemory(){this.asyncMemoize=void 0}},qe=class extends Hi{constructor(t,r){super(t),this.registerDisposer(t),this.downloadFunction=r.get,this.encodeKeyFunction=r.encodeKey??Kt}encodeKeyFunction;downloadFunction;get(t,r){let n=this.encodeKeyFunction(t),i=this.chunks.get(n);return i===void 0&&(i=this.getNewChunk_(ff),i.initialize(n),this.addChunk(i)),i.asyncMemoize===void 0&&(i.asyncMemoize=Yi(async s=>{try{let{data:o,size:a}=await this.downloadFunction(t,s);return i.systemMemoryBytes=a,i.queueManager.updateChunkState(i,z.SYSTEM_MEMORY_WORKER),o}catch(o){throw i.queueManager.updateChunkState(i,z.FAILED),o}})),i.state===z.SYSTEM_MEMORY_WORKER&&i.chunkManager.queueManager.markRecentlyUsed(i),i.asyncMemoize(r)}};function Sv(e,t,r){return e.memoize.get(`simpleAsyncCache:${t}`,()=>new qe(e.addRef(),r))}function ni(e,t,r,n){return e.chunkManager.memoize.get(`getCachedDecodedUrl:${xv(r)}`,()=>{let s=new qe(e.chunkManager.addRef(),{get:async(o,a)=>{let c=await e.kvStoreContext.read(o,{...a,throwIfMissing:!0});try{return r(c,a)}catch(u){throw new Error("Error reading ${url}",{cause:u})}}});return s.registerDisposer(e.addRef()),s}).get(t,n)}var JM=100,jr=class{constructor(t,r){this.base=t,this.format=r}async stat(t){return await this.base.stat(t),{totalSize:void 0}}async read(t){let{byteRange:r}=t;if(r===void 0){let a=await this.base.read(t);return a===void 0?void 0:{response:new Response(Fl(a.response,this.format)),offset:0,length:void 0,totalSize:void 0}}if("suffixLength"in r||r.offset!==0)throw new Error(`Byte range with offset not supported: ${JSON.stringify(r)}`);let n=new Uint8Array(r.length),i=[],s=0,o=r.length+JM;for(;;){let a=await this.base.read({...t,byteRange:{offset:s,length:o-s}});if(a===void 0)return;{let p=new Uint8Array(await a.response.arrayBuffer());i.push(p),s+=p.length}let c=new DecompressionStream("gzip"),u=c.writable.getWriter(),l=[];for(let p of i)l.push(u.write(p));l.push(u.close());let f=c.readable.getReader(),h=0;try{for(;h<n.length;){let{value:p}=await f.read();if(p===void 0)break;let d=n.length-h;p.length>d&&(p=p.subarray(0,d)),n.set(p,h),h+=p.length}if(h===n.length||s===a.totalSize)return h<n.length&&(n=n.subarray(0,h)),{response:new Response(n),offset:0,length:n.length,totalSize:void 0}}finally{await f.cancel(),await Promise.allSettled(l)}o+=Math.min(100,n.length-h)}}getUrl(){return this.base.getUrl()+"|gzip"}};function Ca(e){return e^=e>>>16,e=Math.imul(e,2246822507),e^=e>>>13,e=Math.imul(e,3266489909),e^=e>>>16,e}function wv(e,t){return e<<t|e>>>32-t}function Ev(e,t){let r=e,n=e,i=e,s=e,o=597399067,a=2869860233,c=951274213,u=Math.imul(Number(t>>BigInt(32)),a);u=wv(u,16),u=Math.imul(u,c),n^=u;let l=Math.imul(Number(t&BigInt(4294967295)),o);l=wv(l,15),l=Math.imul(l,a),r^=l;let f=8;return r^=f,n^=f,i^=f,s^=f,r=r+n>>>0,r=r+i>>>0,r=r+s>>>0,n=n+r>>>0,i=i+r>>>0,s=s+r>>>0,r=Ca(r),n=Ca(n),i=Ca(i),s=Ca(s),r=r+n>>>0,r=r+i>>>0,r=r+s>>>0,n=n+r>>>0,BigInt(r)|BigInt(n)<<BigInt(32)}var HM=new Map([[_a.MURMURHASH3_X86_128,e=>Ev(0,e)],[_a.IDENTITY,e=>e]]);function bv(e,t){return t===ri.GZIP&&(e=new jr(e,"gzip")),e}function WM(e,t,r){return new qe(e.addRef(),{encodeKey:n=>n.toString(),get:async(n,i)=>{let s=n&(1n<<BigInt(r.minishardBits))-1n,o=(1n<<BigInt(r.shardBits))-1n&n>>BigInt(r.minishardBits),a=t.path+o.toString(16).padStart(Math.ceil(r.shardBits/4),"0")+".shard",c=new Te(t.store,a),u=BigInt(16)<<BigInt(r.minishardBits),l=s<<4n,f=await Pr(c,{...i,byteRange:{offset:Number(l),length:16},strictByteRange:!0});if(f===void 0)return{data:void 0,size:0};let h=await f.response.arrayBuffer(),p=new DataView(h),d=p.getBigUint64(0,!0),m=p.getBigUint64(8,!0);if(d===m)return{data:void 0,size:0};d+=u,m+=u;let g=await(await Pr(bv(new tt(c,{offset:Number(d),length:Number(m-d)}),r.minishardIndexEncoding),{...i,strictByteRange:!0,throwIfMissing:!0})).response.arrayBuffer();if(g.byteLength%24!==0)throw new Error(`Invalid minishard index length: ${g.byteLength}`);let y=new BigUint64Array(g);l0(y,oe.LITTLE);let I=y.byteLength/24,_=0n,E=u;for(let b=0;b<I;++b){let C=_+y[b];_=y[b]=C;let v=E+y[I+b];y[I+b]=v;let w=y[2*I+b],x=v+w;E=x,y[2*I+b]=x}return{data:{data:y,shardPath:a},size:y.byteLength}}})}function XM(e,t){let r=e.data,n=r.length/3;for(let i=0;i<n;++i){if(r[i]!==t)continue;let s=r[n+i],o=r[2*n+i];return{offset:Number(s),length:Number(o-s)}}}var hf=class extends be{constructor(t,r,n){super(),this.base=r,this.sharding=n,this.minishardIndexCache=this.registerDisposer(WM(t,r,n))}minishardIndexCache;getUrl(t){return`chunk ${t} in ${this.base.store.getUrl(this.base.path)}`}async findKey(t,r){let{sharding:n}=this,o=HM.get(n.hash)(t>>BigInt(n.preshiftBits))&(1n<<BigInt(n.minishardBits+n.shardBits))-1n,a=await this.minishardIndexCache.get(o,r);if(a===void 0)return;let c=XM(a,t);if(c!==void 0)return{minishardEntry:c,shardInfo:{shardPath:a.shardPath,offset:c.offset}}}async readWithShardInfo(t,r){let{sharding:n}=this,i=await this.findKey(t,r);if(i===void 0)return;let{minishardEntry:s,shardInfo:o}=i;return{response:await bv(new tt(new Te(this.base.store,o.shardPath),s),n.dataEncoding).read(r),shardInfo:o}}async stat(t,r){let n=await this.findKey(t,r);if(n===void 0)return;let{sharding:i}=this;return i.dataEncoding!==ri.RAW?{totalSize:void 0}:{totalSize:n.minishardEntry.length}}async read(t,r){let n=await this.readWithShardInfo(t,r);if(n!==void 0)return n.response}get supportsOffsetReads(){return this.sharding.dataEncoding===ri.RAW}get supportsSuffixReads(){return this.sharding.dataEncoding===ri.RAW}};function dn(e,t,r){if(r!==void 0)return e.registerDisposer(new hf(e.chunkManager,t,r))}var fs,_v=0,Cv,Iv={emscripten_notify_memory_growth:e=>{},neuroglancer_draco_receive_decoded_mesh:(e,t,r,n,i)=>{let s=e*3,o=Cv.exports.memory,a=new Uint32Array(o.buffer,r,s).slice(),c=new Uint32Array(o.buffer,n,3*t).slice(),u=new Uint32Array(o.buffer,i,_v+1).slice();fs={indices:a,vertexPositions:c,subChunkOffsets:u}},proc_exit:e=>{throw`proc exit: ${e}`}},pf;function Av(){return pf==null&&(pf=(async()=>{let e=Cv=(await WebAssembly.instantiateStreaming(fetch(new URL("./neuroglancer_draco.wasm",import.meta.url)),{env:Iv,wasi_snapshot_preview1:Iv})).instance;return e.exports._initialize(),e})()),pf}async function Mv(e,t,r){let n=await Av(),i=n.exports.malloc(e.byteLength);new Uint8Array(n.exports.memory.buffer).set(e,i),_v=r?8:1;let o=n.exports.neuroglancer_draco_decode(i,e.byteLength,r,t,!0);if(o===0){let a=fs;if(fs=void 0,a instanceof Error)throw a;return a}throw new Error(`Failed to decode draco mesh: ${o}`)}async function Tv(e){let t=await Av(),r=t.exports.malloc(e.byteLength);new Uint8Array(t.exports.memory.buffer).set(e,r);let i=t.exports.neuroglancer_draco_decode(r,e.byteLength,!1,0,!1);if(i===0){let s=fs;if(fs=void 0,s instanceof Error)throw s;return s.vertexPositions=new Float32Array(s.vertexPositions.buffer),s}throw new Error(`Failed to decode draco mesh: ${i}`)}function kv(e,t,r){let n=new DataView(t),i=n.getUint32(0,!0),s=n.getUint32(4,!0),o=8,a=8+i*4*3;aa(e,t,oe.LITTLE,o,i,a,s),a+=s*4*2;let c=[];for(let u of r.values()){let l=ut[u.dataType]*u.numComponents,f=l*i,h=new Uint8Array(t,a,f);switch(l){case 2:u0(h,oe.LITTLE);break;case 4:case 8:It(h,oe.LITTLE);break}c.push(h),a+=f}e.vertexAttributes=c}var Nv=De("decodeCompresso");async function Dv(e,t,r){let n=await le(Nv,t,[r],new Uint8Array(r));await ft(e,t,n.buffer)}var Pv=De("decodeJxl");async function Rv(e,t,r){let n=e.chunkDataSize,{uint8Array:i}=await le(Pv,t,[r],new Uint8Array(r),n[0]*n[1]*n[2],n[3]||1,1);await lt(e,t,i)}async function Ov(e,t,r){let n=e.chunkDataSize,i=e.source.spec.dataType,{uint8Array:s}=await le(hn,t,[r],new Uint8Array(r),void 0,void 0,n[0]*n[1]*n[2],n[3]||1,ut[i],!1);await ft(e,t,s.buffer)}var ZM=Object.defineProperty,QM=Object.getOwnPropertyDescriptor,ii=(e,t,r,n)=>{for(var i=n>1?void 0:n?QM(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&ZM(t,r,i),i};var eT=!1;function df(e){if(e===void 0)throw new Error("not found");return e}var mn=new Map;mn.set(Gr.RAW,ft);mn.set(Gr.JPEG,Br);mn.set(Gr.COMPRESSED_SEGMENTATION,Xn);mn.set(Gr.COMPRESSO,Dv);mn.set(Gr.PNG,Ov);mn.set(Gr.JXL,Rv);var Uv=class extends ye(ke(Ae),xa){chunkDecoder=mn.get(this.parameters.encoding);kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.sharding);gridShape=(()=>{let e=new Uint32Array(3),{upperVoxelBound:t,chunkDataSize:r}=this.spec;for(let n=0;n<3;++n)e[n]=Math.ceil(t[n]/r[n]);return e})();async download(e,t){let{shardedKvStore:r}=this,n;if(r===void 0){let{kvStore:i}=this,s;{let o=this.computeChunkBounds(e),a=e.chunkDataSize;s=`${i.path}${o[0]}-${o[0]+a[0]}_${o[1]}-${o[1]+a[1]}_${o[2]}-${o[2]+a[2]}`}n=await i.store.read(s,{signal:t})}else{this.computeChunkBounds(e);let{gridShape:i}=this,{chunkGridPosition:s}=e,o=Math.ceil(Math.log2(i[0])),a=Math.ceil(Math.log2(i[1])),c=Math.ceil(Math.log2(i[2])),u=$o(o,a,c,s[0],s[1],s[2]);n=await r.read(u,{signal:t})}n!==void 0&&await this.chunkDecoder(e,t,await n.response.arrayBuffer())}};Uv=ii([G()],Uv);function Aa(e,t){return Yo(e,t,"fragments")}function tT(e,t){let n=new DataView(t).getUint32(0,!0);or(e,Yn(t,oe.LITTLE,4,n))}var Lv=class extends ye(ke(Nt),Sa){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);async download(e,t){let{parameters:r,kvStore:n}=this,i=await Jt(n.store,`${n.path}${e.objectId}:${r.lod}`,{signal:t,throwIfMissing:!0});Aa(e,await i.response.json())}async downloadFragment(e,t){let{kvStore:r}=this,n=await Jt(r.store,`${r.path}${e.fragmentId}`,{signal:t,throwIfMissing:!0});tT(e,await n.response.arrayBuffer())}};Lv=ii([G()],Lv);function rT(e,t){if(t.byteLength<28||t.byteLength%4!==0)throw new Error(`Invalid index file size: ${t.byteLength}`);let r=new DataView(t),n=0,i=B.fromValues(r.getFloat32(n,!0),r.getFloat32(n+4,!0),r.getFloat32(n+8,!0));n+=12;let s=B.fromValues(r.getFloat32(n,!0),r.getFloat32(n+4,!0),r.getFloat32(n+8,!0));n+=12;let o=r.getUint32(n,!0);if(n+=4,t.byteLength<n+(8+4*3)*o)throw new Error(`Invalid index file size for ${o} lods: ${t.byteLength}`);let a=new Float32Array(t,n,o);n+=4*o,It(a,oe.LITTLE);let c=new Float32Array(t,n,o*3);It(c,oe.LITTLE),n+=12*o;let u=new Uint32Array(t,n,o);n+=4*o,It(u,oe.LITTLE);let l=u.reduce((C,v)=>C+v);if(t.byteLength!==n+16*l)throw new Error(`Invalid index file size for ${o} lods and ${l} total fragments: ${t.byteLength}`);let f=new Uint32Array(t,n);It(f,oe.LITTLE);let h=B.fromValues(Number.POSITIVE_INFINITY,Number.POSITIVE_INFINITY,Number.POSITIVE_INFINITY),p=B.fromValues(Number.NEGATIVE_INFINITY,Number.NEGATIVE_INFINITY,Number.NEGATIVE_INFINITY),d=Math.max(1,a.length);{let C=0;for(let v=0;v<o;++v){let w=u[v];if(eT)for(let x=1;x<w;++x){let T=f[C+w*0+(x-1)],M=f[C+w*1+(x-1)],P=f[C+w*2+(x-1)],F=f[C+w*0+x],S=f[C+w*1+x],O=f[C+w*2+x];jn(T,M,P,F,S,O)||console.log(`Fragment index violates zorder constraint: lod=${v}, chunk ${x-1} = [${T},${M},${P}], chunk ${x} = [${F},${S},${O}]`)}for(let x=0;x<3;++x){let T=Number.NEGATIVE_INFINITY,M=Number.POSITIVE_INFINITY,P=C+w*x;for(let F=0;F<w;++F){let S=f[P+F];T=Math.max(T,S),M=Math.min(M,S)}if(w!==0){for(;T>>>d-v-1!==M>>>d-v-1;)++d;v===0&&(h[x]=Math.min(h[x],(1<<v)*M),p[x]=Math.max(p[x],(1<<v)*(T+1)))}}C+=w*4}}let m=0;{let C=0,v=0;for(let w=0;w<o;++w){let x=u[w];m+=C*(w-v),v=w,C=x,m+=x}m+=(d-1-v)*C}let g=new Uint32Array(5*m),y=new Float64Array(m+1),I;{let C=0,v=0,w=0,x=0;for(let T=0;T<o;++T){let M=u[T];for(let P=0;P<M;++P){for(let S=0;S<3;++S)g[5*(v+P)+S]=f[x+P+S*M];let F=f[x+P+3*M];w+=F,y[v+P+1]=w,F===0&&(g[5*(v+P)+4]=2147483648)}for(x+=4*M,T!==0&&w0(g,C,v,v+M),C=v,v+=M;T+1<d&&(T+1>=a.length||a[T+1]===0);){let P=Ho(g,C,v);y.fill(w,v+1,P+1),C=v,v=P,++T}}I=g.slice(0,5*v),e.offsets=y.slice(0,v+1)}let _=e.source,{lodScaleMultiplier:E}=_.parameters.metadata,b=new Float32Array(d);b.set(a,0);for(let C=0;C<a.length;++C)b[C]*=E;e.manifest={chunkShape:i,chunkGridSpatialOrigin:s,clipLowerBound:B.add(h,s,B.multiply(h,h,i)),clipUpperBound:B.add(p,s,B.multiply(p,p,i)),octree:I,lodScales:b,vertexOffsets:c}}async function nT(e,t){let{lod:r}=e,n=e.manifestChunk.source,i=await Mv(new Uint8Array(t),n.parameters.metadata.vertexQuantizationBits,r!==0);Jo(e,i,n.format.vertexPositionFormat)}var Fv=class extends ye(ke(Kn),wa){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.metadata.sharding);async download(e,t){let{shardedKvStore:r}=this,n;if(r===void 0){let{kvStore:s}=this;n=await s.store.read(`${s.path}${e.objectId}.index`,{signal:t})}else({response:n,shardInfo:e.shardInfo}=df(await r.readWithShardInfo(e.objectId,{signal:t})));let i=await df(n).response.arrayBuffer();rT(e,i)}async downloadFragment(e,t){let{kvStore:r}=this,n=e.manifestChunk,i=e.chunkIndex,{shardInfo:s,offsets:o}=n,a=o[i],c=o[i+1],u,l,f;if(s!==void 0){u=s.shardPath;let p=o[o.length-1],d=s.offset-p+a,m=d+c-a;l=d,f=m}else u=`${r.path}${n.objectId}`,l=a,f=c;let h=await Jt(r.store,u,{signal:t,byteRange:{offset:l,length:f-l},throwIfMissing:!0,strictByteRange:!0});await nT(e,await h.response.arrayBuffer())}};Fv=ii([G()],Fv);async function mf(e,t,r){let{shardedKvStore:n}=e;if(n===void 0){let{kvStore:i}=e;return i.store.read(`${i.path}${t}`,{signal:r})}else return n.read(t,{signal:r})}var Bv=class extends ye(ke($r),Ea){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.metadata.sharding);async download(e,t){let{parameters:r}=this,n=df(await mf(this,e.objectId,t));kv(e,await n.response.arrayBuffer(),r.metadata.vertexAttributes)}};Bv=ii([G()],Bv);function Vv(e,t,r){if(e.byteLength<8)throw new Error("Expected at least 8 bytes");let i=new DataView(e);if(i.getUint32(4,!0)!==0)throw new Error("Annotation count too high");let o=i.getUint32(0,!0),a=r.serializedBytes,c=8+a*o,u=o,l=t.type;if(l===Oe.POLYLINE){let E=oT(i,t.rank,o,a,!0);u=E.numInstances,c=E.totalBytes}let f=aT(e,i,o,c),h=new Uint8Array(e,8,c-8),p=new Vn,d=p.typeToInstanceCounts=new Array(br.length);d.fill([]);let{propertyGroupBytes:m}=r;if(m.length>1||l===Oe.POLYLINE){let E=iT(i,h,r,u,o,l,t.rank,!0);p.data=E.outputData,d[Oe.POLYLINE]=E.polylineInstanceCounts}else p.data=h,d[t.type]=Array.from({length:f.length},(E,b)=>b);(p.typeToOffset=new Array(br.length)).fill(0);let y=p.typeToIds=new Array(br.length),I=p.typeToIdMaps=new Array(br.length),_=p.typeToSize=new Array(br.length);return _.fill(0),_[t.type]=u,y.fill([]),y[t.type]=f,I.fill(new Map),I[t.type]=new Map(f.map((E,b)=>[E,b])),p}function iT(e,t,r,n,i,s,o,a){let{propertyGroupBytes:c,serializedBytes:u}=r,l=u*n,f=new Uint8Array(l),h=[];s===Oe.POLYLINE&&(h=sT(e,t,f,r,i,o,a));let p=t;if(s===Oe.POLYLINE&&(p=new Uint8Array(f)),c.length>1){let d=0,m=0;for(let g=0;g<c.length;++g){let y=0,I=c[g];for(let _=0;_<i;++_){let E=1;s===Oe.POLYLINE&&(_===i-1?E=n-h[_]:E=h[_+1]-h[_]);for(let b=0;b<E;++b){let C=d+y*u,v=m+y*I;f.set(p.subarray(C,C+I),v),++y}}d+=I,m+=I*n}}return{outputData:f,polylineInstanceCounts:h}}function sT(e,t,r,n,i,s,o){let a=new DataView(r.buffer),c=0,u=0,l=4,f=s*4,h=n.serializedBytes,p=h-2*f-l,d=8,m=0,g=new Array(i);for(let y=0;y<i;++y){let I=e.getUint32(c+d,o),_=I-1;c+=l;let E=c+I*f;g[y]=m,m+=_;for(let b=0;b<_;++b){let C=b===_-1?1:0,v=b|C<<31;a.setUint32(u,v,o),r.set(t.subarray(c,c+2*f),u+l),r.set(t.subarray(E,E+p),u+l+2*f),c+=f,u+=h}c=E+p}return g}function oT(e,t,r,n,i){let s=8,o=0;for(let a=0;a<r;a++){let c=e.getUint32(s,i),u=c-1,l=c*t*4,f=n-2*t*4;s+=l+f,o+=u}return{totalBytes:s,numInstances:o}}function aT(e,t,r,n){let i=n+8*r;if(e.byteLength!==i)throw new Error(`Expected ${i} bytes, but received: ${e.byteLength} bytes`);let s=n,o=new Array(r);for(let a=0;a<r;++a)o[a]=t.getBigUint64(s+a*8,!0).toString();return o}function cT(e,t,r,n){let i=as[t.type],s=r.serializedBytes,o=new DataView(e),a=0;if(t.type===Oe.POLYLINE){let h=o.getUint32(0,!0)&2147483647,p=r.serializedBytes-(2*4*t.rank+4);s=4+h*4*t.rank+p,a=(h-2)*4*t.rank}let c=t.relationships.length,u=s+4*c;if(e.byteLength<u)throw new Error(`Expected at least ${u} bytes, but received: ${e.byteLength}`);let l=i.deserialize(o,0,!0,t.rank,n,0);r.deserialize(o,a,0,1,!0,l.properties=new Array(t.properties.length)),a=s;let f=l.relatedSegments=[];f.length=c;for(let h=0;h<c;++h){let p=o.getUint32(a,!0);if(e.byteLength<u+p*8)throw new Error(`Expected at least ${u} bytes, but received: ${e.byteLength}`);a+=4;let d=f[h]=new BigUint64Array(p);for(let m=0;m<p;++m)d[m]=o.getBigUint64(a,!0),a+=8}if(a!==e.byteLength)throw new Error(`Expected ${a} bytes, but received: ${e.byteLength}`);return l}var zv=class extends ye(ke(cn),ba){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);shardedKvStore=dn(this,this.kvStore,this.parameters.sharding);async download(e,t){let{shardedKvStore:r}=this,{parent:n}=this,i,{chunkGridPosition:s}=e;if(r===void 0){let{kvStore:o}=this,a=`${o.path}${s.join("_")}`;i=await o.store.read(a,{signal:t})}else{let{upperChunkBound:o}=this.spec,{chunkGridPosition:a}=e,c=r0(a,o);i=await r.read(c,{signal:t})}i!==void 0&&(e.data=Vv(await i.response.arrayBuffer(),n.parameters,n.annotationPropertySerializer))}};zv=ii([G()],zv);var $v=class extends ye(ke(Gn),Ia){kvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.byId.url);shardedKvStore=dn(this,this.kvStore,this.parameters.byId.sharding);relationshipIndexSource=this.parameters.relationships.map(e=>{let t=this.sharedKvStoreContext.kvStoreContext.getKvStore(e.url),r=dn(this,t,e.sharding);return{kvStore:t,shardedKvStore:r}});annotationPropertySerializer=new os(this.parameters.rank,as[this.parameters.type].serializedBytes(this.parameters.rank),this.parameters.properties);async downloadSegmentFilteredGeometry(e,t,r){let n=await mf(this.relationshipIndexSource[t],e.objectId,r);n!==void 0&&(e.data=Vv(await n.response.arrayBuffer(),this.parameters,this.annotationPropertySerializer))}async downloadMetadata(e,t){let r=BigInt(e.key),n=await mf(this,r,t);n===void 0?e.annotation=null:e.annotation=cT(await n.response.arrayBuffer(),this.parameters,this.annotationPropertySerializer,e.key)}};$v=ii([G()],$v);var uT=Object.defineProperty,lT=Object.getOwnPropertyDescriptor,yf=(e,t,r,n)=>{for(var i=n>1?void 0:n?lT(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&uT(t,r,i),i};function fT(e,t,r){if(t&&t.charAt(0)==="~"){let n=t.substring(1).split(":"),i={offset:Number(n[1]),length:Number(n[2])};return Jt(e.store,`${e.path}initial/${n[0]}`,{signal:r,byteRange:i,throwIfMissing:!0})}return Jt(e.store,`${e.path}dynamic/${t}`,{signal:r,throwIfMissing:!0})}function hT(e,t,r,n){return r.sharding?fT(e,t,n):Jt(e.store,`${e.path}/${t}`,{signal:n,throwIfMissing:!0})}async function pT(e,t){let r=await Tv(t);or(e,r)}var Gv=class extends ye(ke(Nt),va){manifestRequestCount=new Map;newSegments=new $n;manifestHttpSource=lf(this.sharedKvStoreContext.kvStoreContext,this.parameters.manifestUrl);fragmentKvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.fragmentUrl);addNewSegment(e){let{newSegments:t}=this;t.add(e);let r=1e3*60*10;setTimeout(()=>{t.delete(e)},r)}async download(e,t){let{parameters:r,newSegments:n,manifestRequestCount:i}=this;if(uf(e.objectId,r.nBitsForLayerId))return Aa(e,{fragments:[]});let{fetchOkImpl:s,baseUrl:o}=this.manifestHttpSource,a=`/manifest/${e.objectId}:${r.lod}?verify=1&prepend_seg_ids=1`,c=await(await s(o+a,{signal:t})).json(),u=a;if(n.has(e.objectId)){let l=(i.get(u)??0)+1;i.set(u,l),setTimeout(()=>{this.chunkManager.queueManager.updateChunkState(e,z.QUEUED)},2**l*1e3)}else i.delete(u);return Aa(e,c)}async downloadFragment(e,t){let{response:r}=await hT(this.fragmentKvStore,e.fragmentId,this.parameters,t);await pT(e,new Uint8Array(await r.arrayBuffer()))}getFragmentKey(e,t){return pv(t)}};Gv=yf([G()],Gv);var gf=class extends fe{chunkGridPosition;source=null;segment;leaves=new BigUint64Array(0);chunkDataSize;initializeVolumeChunk(t,r){super.initialize(t),this.chunkGridPosition=Float32Array.from(r)}initializeChunkedGraphChunk(t,r,n){this.initializeVolumeChunk(t,r),this.chunkDataSize=null,this.systemMemoryBytes=16,this.gpuMemoryBytes=0,this.segment=n}downloadSucceeded(){this.systemMemoryBytes=16,this.systemMemoryBytes+=this.leaves.byteLength,this.queueManager.updateChunkState(this,z.SYSTEM_MEMORY_WORKER),this.priorityTier<se.RECENT&&this.source.chunkManager.scheduleUpdateChunkPriorities(),super.downloadSucceeded()}freeSystemMemory(){this.leaves=new BigUint64Array(0)}};function dT(e){return BigUint64Array.from(e,yr)}var jv=class extends ye(ke(Ne),ya){spec;tempChunkDataSize;tempChunkPosition;httpSource=lf(this.sharedKvStoreContext.kvStoreContext,this.parameters.url);constructor(e,t){super(e,t),this.spec=t.spec;let r=this.spec.rank;this.tempChunkDataSize=new Uint32Array(r),this.tempChunkPosition=new Float32Array(r)}async download(e,t){let r=this.computeChunkBounds(e),n=e.chunkDataSize,i=`${r[0]}-${r[0]+n[0]}_${r[1]}-${r[1]+n[1]}_${r[2]}-${r[2]+n[2]}`,{fetchOkImpl:s,baseUrl:o}=this.httpSource,a=s(`${o}/${e.segment}/leaves?int64_as_str=1&bounds=${i}`,{signal:t});await this.withErrorMessage(a,`Fetching leaves of segment ${e.segment} in region ${i}: `).then(c=>c.json()).then(c=>{e.leaves=dT(c.leaf_ids)}).catch(c=>{c instanceof Error&&c.name==="AbortError"||console.error(c)})}getChunk(e,t){let r=`${tr(e)}-${t}`,n=this.chunks.get(r);return n===void 0&&(n=this.getNewChunk_(gf),n.initializeChunkedGraphChunk(r,e,t),this.addChunk(n)),n}computeChunkBounds(e){return Vl(this,e)}async withErrorMessage(e,t){return e.catch(async r=>{if(r instanceof bt&&r.response){let n=await yv(r);throw new Error(`[${r.response.status}] ${t}${n??""}`)}throw r})}};jv=yf([G()],jv);var qv=B.create(),mT=B.create(),gT=B.create(),Kv=class extends an(Et(Xe(ir))){source;localPosition;leafRequestsActive;nBitsForLayerId;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.localPosition=e.get(t.localPosition),this.leafRequestsActive=e.get(t.leafRequestsActive),this.nBitsForLayerId=e.get(t.nBitsForLayerId),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities(),this.debouncedupdateDisplayState()}))}attach(e){let t=()=>this.chunkManager.scheduleUpdateChunkPriorities(),{view:r}=e;e.registerDisposer(t),e.registerDisposer(r.projectionParameters.changed.add(t)),e.registerDisposer(r.visibility.changed.add(t)),e.state={displayDimensionRenderInfo:r.projectionParameters.value.displayDimensionRenderInfo}}get renderRatioLimit(){return gv}updateChunkPriorities(){let{source:e,chunkManager:t}=this;t.registerLayer(this);for(let r of this.attachments.values()){let{view:n}=r,i=n.visibility.value;if(i===Number.NEGATIVE_INFINITY)continue;let s=r.state,{transformedSource:o}=s,a=n.projectionParameters.value;if(!o)continue;let c=a.pixelSize*1.1,u=o.effectiveVoxelSize;if(this.leafRequestsActive.value=this.renderRatioLimit>=c/Math.min(...u),!this.leafRequestsActive.value)continue;let l=Ge(i),f=je(i),{chunkLayout:h}=o,{size:p,finiteRank:d}=h,m=gT,g=mT;B.copy(m,p);for(let I=d;I<3;++I)m[I]=0,g[I]=0;let{centerDataPosition:y}=a;h.globalToLocalSpatial(g,y),To(a,this.localPosition.value,o,ko(a,h),I=>{B.multiply(qv,I,m);let _=-B.distance(g,qv),{curPositionInChunks:E}=o;xr(this,(b,C)=>{if(uf(b,this.nBitsForLayerId.value))return;let v=e.getChunk(E,b);t.requestChunk(v,l,f+_,z.SYSTEM_MEMORY_WORKER),++this.numVisibleChunksNeeded,v.state===z.GPU_MEMORY&&++this.numVisibleChunksAvailable})})}}forEachSelectedRootWithLeaves(e){let{source:t}=this;for(let r of t.chunks.values())r.state===z.SYSTEM_MEMORY_WORKER&&r.priorityTier<se.RECENT&&this.visibleSegments.has(r.segment)&&r.leaves.length&&e(r.segment,r.leaves)}debouncedupdateDisplayState=tn(()=>{this.updateDisplayState()},100);updateDisplayState(){let e=new Map,t=new Map;this.forEachSelectedRootWithLeaves((r,n)=>{t.set(r,(t.get(r)??0)+n.length)}),this.forEachSelectedRootWithLeaves((r,n)=>{e.has(r)||(e.set(r,new $n),e.get(r).reserve(t.get(r)),e.get(r).add(r)),e.get(r).add(n)});for(let[r,n]of e){let i=[...n].filter(s=>!this.segmentEquivalences.has(s));for(let s of i)this.segmentEquivalences.link(r,s)}}};Kv=yf([G(dv)],Kv);X(mv,function(e){let t=this.get(e.view),r=this.get(e.layer),n=r.attachments.get(t);n.state.transformedSource=sn(this,e.sources,r)[0][0],n.state.displayDimensionRenderInfo=e.displayDimensionRenderInfo,r.chunkManager.scheduleUpdateChunkPriorities()});X(hv,function(e){this.get(e.rpcId).addNewSegment(e.segment)});var Ma=De("decodeBlosc");var qr=De("decodeZstd");var si=(e=>(e[e.RAW=0]="RAW",e[e.ZLIB=1]="ZLIB",e[e.GZIP=2]="GZIP",e[e.BLOSC=3]="BLOSC",e[e.ZSTD=4]="ZSTD",e))(si||{}),Ta=class{url;encoding;static RPC_ID="n5/VolumeChunkSource"};var yT=Object.defineProperty,vT=Object.getOwnPropertyDescriptor,xT=(e,t,r,n)=>{for(var i=n>1?void 0:n?vT(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&yT(t,r,i),i};async function ST(e,t,r,n){let i=new DataView(r),s=i.getUint16(0,!1);if(s!==0)throw new Error(`Unsupported mode: ${s}.`);let o=i.getUint16(2,!1);if(o!==e.source.spec.rank)throw new Error("Number of dimensions must be 3.");let a=4,c=new Uint32Array(o);for(let l=0;l<o;++l)c[l]=i.getUint32(a,!1),a+=4;e.chunkDataSize=c;let u=new Uint8Array(r,a);switch(n){case si.ZLIB:u=new Uint8Array(await wr(u,"deflate"));break;case si.GZIP:u=new Uint8Array(await wr(u,"gzip"));break;case si.BLOSC:u=await le(Ma,t,[u.buffer],u);break;case si.ZSTD:u=await le(qr,t,[u.buffer],u);break}await ft(e,t,u.buffer,oe.BIG,u.byteOffset,u.byteLength)}var Yv=class extends ye(ke(Ae),Ta){chunkKvStore=this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);async download(e,t){let{parameters:r,chunkKvStore:n}=this,{chunkGridPosition:i}=e,s=n.path,o=this.spec.rank;for(let c=0;c<o;++c)c!==0&&(s+="/"),s+=`${i[c]}`;let a=await n.store.read(s,{signal:t});a!==void 0&&await ST(e,t,await a.response.arrayBuffer(),r.encoding)}};Yv=xT([G()],Yv);var Ke=Ri(Rx(),1);var Ox="nifti/getNiftiVolumeInfo",Xa=class{url;static RPC_ID="nifti/VolumeChunkSource"};var ek=Object.defineProperty,tk=Object.getOwnPropertyDescriptor,rk=(e,t,r,n)=>{for(var i=n>1?void 0:n?tk(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&ek(t,r,i),i};var Kf=class{uncompressedData;header};async function nk(e,t){let r=await e.response.arrayBuffer();(0,Ke.isCompressed)(r)&&(r=await wr(r,"gzip",t.signal));let n=new Kf;n.uncompressedData=r;let i=(0,Ke.readHeader)(r);if(i===null)throw new Error("Failed to parse NIFTI header.");return n.header=i,{data:n,size:r.byteLength}}function Lx(e,t,r){return ni(e,t,nk,r)}async function ik(e,t,r){return(await Lx(e,t,r)).header}function sk(e){return ge.fromValues(e[0][0],e[1][0],e[2][0],e[3][0],e[0][1],e[1][1],e[2][1],e[3][1],e[0][2],e[1][2],e[2][2],e[3][2],e[0][3],e[1][3],e[2][3],e[3][3])}var Fx=(e=>(e[e.NONE=0]="NONE",e[e.BINARY=1]="BINARY",e[e.UINT8=2]="UINT8",e[e.INT16=4]="INT16",e[e.INT32=8]="INT32",e[e.FLOAT32=16]="FLOAT32",e[e.COMPLEX64=32]="COMPLEX64",e[e.FLOAT64=64]="FLOAT64",e[e.RGB24=128]="RGB24",e[e.INT8=256]="INT8",e[e.UINT16=512]="UINT16",e[e.UINT32=768]="UINT32",e[e.INT64=1024]="INT64",e[e.UINT64=1280]="UINT64",e[e.FLOAT128=1536]="FLOAT128",e[e.COMPLEX128=1792]="COMPLEX128",e[e.COMPLEX256=2048]="COMPLEX256",e))(Fx||{}),ok=new Map([[256,{dataType:W.INT8}],[2,{dataType:W.UINT8}],[4,{dataType:W.INT16}],[512,{dataType:W.UINT16}],[8,{dataType:W.INT32}],[768,{dataType:W.UINT32}],[1024,{dataType:W.UINT64}],[1280,{dataType:W.UINT64}],[16,{dataType:W.FLOAT32}]]);xt(Ox,async function(e,t){let r=this.get(e.sharedKvStoreContext),n=await ik(r,e.url,t),i=ok.get(n.datatypeCode);if(i===void 0)throw new Error(`Unsupported data type: ${Fx[n.datatypeCode]||n.datatypeCode}.`);let s=1,o="";switch(n.xyzt_units&Ke.NIFTI1.SPATIAL_UNITS_MASK){case Ke.NIFTI1.UNITS_METER:s=1,o="m";break;case Ke.NIFTI1.UNITS_MM:s=1e3,o="m";break;case Ke.NIFTI1.UNITS_MICRON:s=1e6,o="m";break}let a="",c=1;switch(n.xyzt_units&Ke.NIFTI1.TEMPORAL_UNITS_MASK){case Ke.NIFTI1.UNITS_SEC:a="s",c=1;break;case Ke.NIFTI1.UNITS_MSEC:a="s",c=1e3;break;case Ke.NIFTI1.UNITS_USEC:a="s",c=1e6;break;case Ke.NIFTI1.UNITS_HZ:a="Hz",c=1;break;case Ke.NIFTI1.UNITS_RADS:a="rad/s",c=1;break}let u=[o,o,o,a,"","",""],l=Float64Array.of(n.pixDims[1]/s,n.pixDims[2]/s,n.pixDims[3]/s,n.pixDims[4]/c,n.pixDims[5],n.pixDims[6],n.pixDims[7]),f=Float64Array.of(1/s,1/s,1/s,1/c,1,1,1),h=["i","j","k","m","c^","c1^","c2^"],p=["x","y","z","t","c^","c1^","c2^"],d=n.dims[0];h=h.slice(0,d),p=p.slice(0,d),u=u.slice(0,d),l=l.slice(0,d),f=f.slice(0,d);let{quatern_b:m,quatern_c:g,quatern_d:y}=n,I=Math.sqrt(1-m*m-g*g-y*y),_=n.pixDims[0]===-1?-1:1,E=B.fromValues(n.qoffset_x,n.qoffset_y,n.qoffset_z),b=sk(n.affine),C=ig(ge.create(),E,Qt.fromValues(m,g,y,I),eg,_),v=nl(Float64Array,d+1),w=Math.min(3,d);for(let T=0;T<w;++T){for(let M=0;M<w;++M)v[M*(d+1)+T]=C[M*4+T];v[d*(d+1)+T]=C[12+T]}return{value:{rank:d,sourceNames:h,viewNames:p,units:u,sourceScales:l,viewScales:f,description:n.description,transform:v,dataType:i.dataType,volumeSize:Uint32Array.from(n.dims.slice(1,1+d))}}});var Ux=class extends ye(ke(Ae),Xa){async download(e,t){e.chunkDataSize=this.spec.chunkDataSize;let r=await Lx(this.sharedKvStoreContext,this.parameters.url,{signal:t}),n=(0,Ke.readImage)(r.header,r.uncompressedData);await ft(e,t,n,r.header.littleEndian?oe.LITTLE:oe.BIG)}};Ux=rk([G()],Ux);var Bx=De("parseOBJFromArrayBuffer");var zx="single_mesh/SingleMeshLayer",$x="single_mesh/getSingleMeshInfo",Vx="",Yf=class{meshSourceUrl},Za=class extends Yf{info;static RPC_ID="single_mesh/SingleMeshSource"};var ak=Object.defineProperty,ck=Object.getOwnPropertyDescriptor,qx=(e,t,r,n)=>{for(var i=n>1?void 0:n?ck(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&ak(t,r,i),i};var uk=50,Jf=class extends fe{data=null;freeSystemMemory(){this.data=null}serialize(t,r){super.serialize(t,r);let{vertexPositions:n,indices:i,vertexNormals:s,vertexAttributes:o}=this.data;t.vertexPositions=n,t.indices=i,t.vertexNormals=s,t.vertexAttributes=o;let a=new Set;a.add(n.buffer),a.add(i.buffer),a.add(s.buffer);for(let c of o)a.add(c.buffer);r.push(...a),this.data=null}downloadSucceeded(){let{vertexPositions:t,indices:r,vertexNormals:n,vertexAttributes:i}=this.data,s=this.gpuMemoryBytes=t.byteLength+r.byteLength+n.byteLength;for(let o of i)s+=o.byteLength;this.systemMemoryBytes=this.gpuMemoryBytes=s,super.downloadSucceeded()}},Kx=new Map;function Qa(e,t){Kx.set(e,t)}var lk=/^(?:([a-zA-Z-+_]+):\/\/)?(.*)$/;function fk(e,t){let r=t.match(lk);if(r===null||r[1]===void 0)throw new Error('Data source URL must have the form "<protocol>://<path>".');let n=r[1],i=e.get(n);if(i===void 0)throw new Error(`Unsupported data source: ${JSON.stringify(n)}.`);return[i,r[2],n]}function hk(e,t,r){let[n,i]=fk(Kx,t);return n.getMesh(e,i,r)}function Yx(e,t,r){return hk(e,t.meshSourceUrl,r)}var Gx=class extends ye(ke(Ne),Za){getChunk(){let e=Vx,t=this.chunks.get(e);return t===void 0&&(t=this.getNewChunk_(Jf),t.initialize(e),this.addChunk(t)),t}async download(e,t){let r=await Yx(this.sharedKvStoreContext,this.parameters,{signal:t});if(Kt(r.info)!==Kt(this.parameters.info))throw new Error("Mesh info has changed.");r.vertexNormals===void 0&&(r.vertexNormals=Rl(r.vertexPositions,r.indices)),e.data=r}};Gx=qx([G()],Gx);var pk=Et(Xe(me)),jx=class extends pk{source;constructor(e,t){super(e,t),this.source=this.registerDisposer(e.getRef(t.source)),this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(()=>{this.updateChunkPriorities()}))}updateChunkPriorities(){let e=this.visibility.value;if(e===Number.NEGATIVE_INFINITY)return;let t=Ge(e),r=je(e),{source:n,chunkManager:i}=this,s=n.getChunk();i.requestChunk(s,t,r+uk)}};jx=qx([G(zx)],jx);xt($x,async function(e,t){let r=this.get(e.sharedKvStoreContext),n=e.parameters;return{value:(await Yx(r,n,t)).info}});async function dk(e,t){let r=await e.response.arrayBuffer();return le(Bx,t.signal,[r],r)}Qa("obj",{description:"OBJ",getMesh:(e,t,r)=>ni(e,t,dk,r)});var Ss=!1;var Hf=class{baseUrl;owner;project;stack;channel},Wf=class extends Hf{renderArgs},ec=class extends Wf{dims;level;encoding;static RPC_ID="render/TileChunkSource"};var mk=Object.defineProperty,gk=Object.getOwnPropertyDescriptor,yk=(e,t,r,n)=>{for(var i=n>1?void 0:n?gk(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&mk(t,r,i),i};var ws=new Map;ws.set("jpg",async(e,t,r)=>{let n=e.chunkDataSize,{uint8Array:i}=await le(Hn,t,[r],new Uint8Array(r),void 0,void 0,n[0]*n[1]*n[2],3,!0);await lt(e,t,i)});ws.set("png",async(e,t,r)=>{let n=e.chunkDataSize,{uint8Array:i}=await le(hn,t,[r],new Uint8Array(r),n[0],n[1],n[0]*n[1]*n[2],4,1,!1);await lt(e,t,i)});ws.set("png16",async(e,t,r)=>{let n=e.chunkDataSize,{uint8Array:i}=await le(hn,t,[r],new Uint8Array(r),n[0],n[1],n[0]*n[1]*n[2],1,2,!1);await lt(e,t,i)});ws.set("raw16",(e,t,r)=>ft(e,t,r,oe.BIG));var Jx=class extends ye(Ae,ec){chunkDecoder=ws.get(this.parameters.encoding);queryString=(()=>{let{parameters:e}=this,t=new URLSearchParams;e.channel!==void 0&&t.append("channel",e.channel);for(let[r,n]of Object.entries(e.renderArgs))t.append(r,n);return t.toString()})();async download(e,t){let{parameters:r}=this,{chunkGridPosition:n}=e,i=1/2**r.level;e.chunkDataSize=this.spec.chunkDataSize;let s=e.chunkDataSize[0]*2**r.level,o=e.chunkDataSize[1]*2**r.level,a=B.create();a[0]=n[0]*s,a[1]=n[1]*o,a[2]=n[2];let c;r.encoding==="raw16"?c="raw16-image":r.encoding==="png16"?c="png16-image":r.encoding==="png"?c="png-image":c="jpeg-image";let u=`/render-ws/v1/owner/${r.owner}/project/${r.project}/stack/${r.stack}/z/${a[2]}/box/${a[0]},${a[1]},${s},${o},${i}/${c}`,l=await Re(`${r.baseUrl}${u}?${this.queryString}`,{signal:t});await this.chunkDecoder(e,t,await l.arrayBuffer())}};Jx=yk([G()],Jx);var Hx=De("parseVTKFromArrayBuffer");async function vk(e,t){let r=await e.response.arrayBuffer();return le(Hx,t.signal,[r],r)}Qa("vtk",{description:"VTK",getMesh:async(e,t,r)=>{let n=await ni(e,t,vk,r),i={info:{numTriangles:n.numTriangles,numVertices:n.numVertices,vertexAttributes:[]},indices:n.indices,vertexPositions:n.vertexPositions,vertexAttributes:[]};for(let s of n.vertexAttributes)i.info.vertexAttributes.push({name:s.name,dataType:W.FLOAT32,numComponents:s.numComponents}),i.vertexAttributes.push(s.data);return i}});var ne=(e=>(e[e.arrayToArray=0]="arrayToArray",e[e.arrayToBytes=1]="arrayToBytes",e[e.bytesToBytes=2]="bytesToBytes",e))(ne||{});var xi={[ne.arrayToArray]:new Map,[ne.arrayToBytes]:new Map,[ne.bytesToBytes]:new Map,sharding:new Map};function pt(e){e.kind===ne.arrayToBytes&&"getShardedKvStore"in e?xi.sharding.set(e.name,e):xi[e.kind].set(e.name,e)}async function tc(e,t,r){let n=e[ne.bytesToBytes];for(let o=n.length;o--;){let a=n[o],c=xi[ne.bytesToBytes].get(a.name);if(c===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(a.name)}`);t=await c.decode(a.configuration,t,r)}let i;{let o=e[ne.arrayToBytes],a=xi[ne.arrayToBytes].get(o.name);if(a===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(o.name)}`);i=await a.decode(o.configuration,e.arrayInfo[e.arrayInfo.length-1],t,r)}let s=e[ne.arrayToArray];for(let o=s.length;o--;){let a=s[o],c=xi[ne.arrayToArray].get(a.name);if(c===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(a.name)}`);i=await c.decode(a.configuration,e.arrayInfo[o],i,r)}return i}function Wx(e,t,r){let n=r.store,i=t;for(;;){let{shardingInfo:c}=i;if(c===void 0)break;let u=i[ne.arrayToBytes],l=xi.sharding.get(u.name);if(l===void 0)throw new Error(`Unsupported codec: ${JSON.stringify(u.name)}`);n=l.getShardedKvStore(u.configuration,e,n),i=c.subChunkCodecs}let s=i,o=r.path;function a(c,u){let l=o+u,f=c.length,h=t;for(;h.shardingInfo!==void 0;){let p=t.layoutInfo[t.layoutInfo.length-1],{physicalToLogicalDimension:d,readChunkShape:m}=p,{subChunkShape:g,subChunkGridShape:y,subChunkCodecs:I}=h.shardingInfo,_=new Array(f);for(let E=0;E<f;++E){let b=d[f-1-E];_[b]=Math.floor(c[E]*m[b]/g[b])%y[b]}l={base:l,subChunk:_},h=I}return l}return{kvStore:n,getChunkKey:a,decodeCodecs:s}}pt({name:"blosc",kind:ne.bytesToBytes,decode(e,t,r){return le(Ma,r,[t.buffer],t)}});pt({name:"zstd",kind:ne.bytesToBytes,decode(e,t,r){return le(qr,r,[t.buffer],t)}});pt({name:"bytes",kind:ne.arrayToBytes,async decode(e,t,r,n){let{dataType:i,chunkShape:s}=t,o=s.reduce((l,f)=>l*f,1),a=ut[i],c=o*a;if(r.byteLength!==c)throw new Error(`Raw-format chunk is ${r.byteLength} bytes, but ${o} * ${a} = ${c} bytes are expected.`);let u=Co(i,r.buffer,r.byteOffset,r.byteLength);return qn(u,e.endian,a),u}});var Xf=4;pt({name:"crc32c",kind:ne.bytesToBytes,async decode(e,t,r){if(t.length<Xf)throw new Error(`Expected buffer of size at least ${Xf} bytes but received: ${t.length} bytes`);return t.subarray(0,t.length-Xf)}});var rc=class{url;metadata;static RPC_ID="zarr/VolumeChunkSource"};for(let[e,t]of[["gzip","gzip"],["zlib","deflate"]])pt({name:e,kind:ne.bytesToBytes,async decode(r,n,i){return new Uint8Array(await wr(n,t,i))}});function Zf(e,t,r){Ee(e);let n=re(e,"name",s=>t(et(s))),i=re(e,"configuration",s=>(s===void 0?s={}:Ee(s),r(s,n)));return{name:n,configuration:i}}function xk(e){let{name:t,configuration:r}=Zf(e,n=>{let i=Xx.get(n);if(i===void 0)throw new Error(`Unknown codec: ${JSON.stringify(n)}`);return i},n=>n);return{resolver:t,configuration:r}}var Xx=new Map;function Zx(e){Xx.set(e.name,e)}function nc(e,t){let r=[],n=[],i=[],s=[];n.push(t);let o=ct(e,xk),a=o.length,c=0;for(;c<a;++c){let{resolver:m,configuration:g}=o[c];if(m.kind!==ne.arrayToArray)break;let y=m,{configuration:I,encodedArrayInfo:_}=y.resolve(g,t);n.push(_),t=_,r.push({kind:ne.arrayToArray,name:m.name,configuration:I})}if(c===a||o[c].resolver.kind!==ne.arrayToBytes)throw new Error("Missing array -> bytes codec");let{codecSpec:u,layoutInfo:l,encodedSize:f,shardingInfo:h}=(()=>{let{resolver:m,configuration:g}=o[c],y=m,{configuration:I,shardingInfo:_,encodedSize:E}=y.resolve(g,t);if(_!==void 0&&c+1!==a)throw new Error("bytes -> bytes codecs not supported following sharding codec");let b=y.getDecodedArrayLayoutInfo(I,t);return{codecSpec:{name:m.name,kind:ne.arrayToBytes,configuration:I},layoutInfo:b,encodedSize:E,shardingInfo:_}})();i[c]=l,s.push(f);let p=f,d=[];for(++c;c<a;){let{resolver:m,configuration:g}=o[c];if(m.kind!==ne.bytesToBytes)throw new Error(`Expected bytes -> bytes codec, but received ${JSON.stringify(m.name)} of kind ${ne[m.kind]}`);let y=m,{configuration:I,encodedSize:_}=y.resolve(g,p);d.push({name:m.name,kind:m.kind,configuration:I}),s.push(_),++c}for(let m=r.length-1;m>=0;--m)i[m]=o[m].resolver.getDecodedArrayLayoutInfo(r[m].configuration,n[m],i[m+1]);return{[ne.arrayToArray]:r,[ne.arrayToBytes]:u,[ne.bytesToBytes]:d,arrayInfo:n,layoutInfo:i,shardingInfo:h,encodedSize:s}}var ic=(e=>(e[e.DEFAULT=0]="DEFAULT",e[e.V2=1]="V2",e))(ic||{});function e1(e,t){return We(new Array(t),e,r=>{if(typeof r!="number"||!Number.isInteger(r)||r<=0)throw new Error(`Expected positive integer, but received: ${JSON.stringify(r)}`);return r})}var Qx=new Map([["",{unit:"",scale:1}],["angstrom",{unit:"m",scale:1e-10}],["foot",{unit:"m",scale:.3048}],["inch",{unit:"m",scale:.0254}],["mile",{unit:"m",scale:1609.34}],["parsec",{unit:"m",scale:0x6da012f95c9e88}],["yard",{unit:"m",scale:.9144}],["minute",{unit:"s",scale:60}],["hour",{unit:"s",scale:60*60}],["day",{unit:"s",scale:60*60*24}]]);for(let e of["meter","second"])for(let t of sl){let{longPrefix:r,prefix:n}=t;if(r===void 0)continue;let i={unit:e[0],scale:10**t.exponent};Qx.set(`${r}${e}`,i),Qx.set(`${n}${e[0]}`,i)}var Es=(e=>(e[e.START=0]="START",e[e.END=1]="END",e))(Es||{});Zx({name:"sharding_indexed",kind:ne.arrayToBytes,resolve(e,t){Ee(e);let r=re(e,"chunk_shape",c=>e1(c,t.chunkShape.length)),n=nr(e,"index_location",c=>Tn(c,Es,/^[a-z]+$/),1),i=Array.from(t.chunkShape,(c,u)=>{let l=r[u];if(c%l!==0)throw new Error(`sub-chunk shape of ${JSON.stringify(l)} does not evenly divide outer chunk shape of ${JSON.stringify(t.chunkShape)}`);return c/l}),s=Array.from(i);s.push(2);let o=re(e,"index_codecs",c=>nc(c,{dataType:W.UINT64,chunkShape:s}));if(o.encodedSize[o.encodedSize.length-1]===void 0)throw new Error("index_codecs must specify fixed-size encoding");let a=re(e,"codecs",c=>nc(c,{dataType:t.dataType,chunkShape:r}));return{configuration:{indexCodecs:o,subChunkCodecs:a,subChunkShape:r,subChunkGridShape:i,indexLocation:n},shardingInfo:{subChunkShape:r,subChunkGridShape:i,subChunkCodecs:a}}},getDecodedArrayLayoutInfo(e,t){return e.subChunkCodecs.layoutInfo[0]}});var t1=BigInt("18446744073709551615");function Sk(e,t,r){return new qe(e.addRef(),{get:async(n,i)=>{let{indexCodecs:s}=r,o=s.encodedSize[s.encodedSize.length-1],a;switch(r.indexLocation){case Es.START:a={offset:0,length:o};break;case Es.END:a={suffixLength:o};break}let c=await t.read(n,{...i,byteRange:a});if(c===void 0)return{size:0,data:void 0};let u=await tc(r.indexCodecs,new Uint8Array(await c.response.arrayBuffer()),i.signal);return{size:u.byteLength,data:new BigUint64Array(u.buffer,u.byteOffset,u.byteLength/8)}}})}var Qf=class extends be{constructor(t,r,n){super(),this.configuration=t,this.base=n,this.indexCache=this.registerDisposer(Sk(r,n,t));let{subChunkGridShape:i}=this.configuration,s=i.length,o=this.configuration.indexCodecs.layoutInfo[0].physicalToLogicalDimension,a=this.indexStrides=new Array(s+1),c=1;for(let u=s;u>=0;--u){let l=o[u];a[l]=c,c*=l===s?2:i[l]}}indexCache;indexStrides;async findKey(t,r){let n=await this.indexCache.get(t.base,r);if(n===void 0)return;let i=this.configuration.subChunkShape.length,{subChunk:s}=t,{indexStrides:o}=this,a=0;for(let l=0;l<i;++l){let f=s[l];a+=f*o[l]}let c=n[a],u=n[a+o[i]];if(!(c===t1&&u===t1))return{offset:Number(c),length:Number(u)}}async stat(t,r){let n=await this.findKey(t,r);if(n!==void 0)return{totalSize:n.length}}async read(t,r){let n=await this.findKey(t,r);if(n!==void 0)return new tt(new Te(this.base,t.base),n).read(r)}getUrl(t){return`subchunk ${JSON.stringify(t.subChunk)} within shard ${this.base.getUrl(t.base)}`}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};pt({name:"sharding_indexed",kind:ne.arrayToBytes,getShardedKvStore(e,t,r){return new Qf(e,t,r)}});pt({name:"transpose",kind:ne.arrayToArray,async decode(e,t,r,n){return r}});var wk=Object.defineProperty,Ek=Object.getOwnPropertyDescriptor,bk=(e,t,r,n)=>{for(var i=n>1?void 0:n?Ek(t,r):t,s=e.length-1,o;s>=0;s--)(o=e[s])&&(i=(n?o(t,r,i):o(i))||i);return n&&i&&wk(t,r,i),i};var r1=class extends ye(ke(Ae),rc){chunkKvStore=Wx(this.chunkManager,this.parameters.metadata.codecs,this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url));async download(e,t){e.chunkDataSize=this.spec.chunkDataSize;let{parameters:r}=this,{chunkGridPosition:n}=e,{metadata:i}=r,s="",o=this.spec.rank,{physicalToLogicalDimension:a}=i.codecs.layoutInfo[0],c;i.chunkKeyEncoding===ic.DEFAULT?(s+="c",c=i.dimensionSeparator):(c="",o===0&&(s+="0"));let u=new Array(o),{readChunkShape:l}=i.codecs.layoutInfo[0],{chunkShape:f}=i;for(let d=0;d<o;++d){let m=a[o-1-d];u[m]=Math.floor(n[d]*l[m]/f[m])}for(let d=0;d<o;++d)s+=`${c}${u[d]}`,c=i.dimensionSeparator;let{chunkKvStore:h}=this,p=await h.kvStore.read(h.getChunkKey(n,s),{signal:t});if(p!==void 0){let d=await tc(h.decodeCodecs,new Uint8Array(await p.response.arrayBuffer()),t);await lt(e,t,d)}}};r1=bk([G()],r1);function n1(e){let t=e.match(/^([0-9]+)-([0-9]+)$/);if(t!==null){let r=Number(t[1]),n=Number(t[2]);if(n>=r)return{offset:r,length:n-r}}throw new Error(`Invalid key ${JSON.stringify(e)} for "byte-range:", expected "<begin>-<end>"`)}var sc=class{constructor(t){this.base=t}getUrl(t){return this.base.getUrl()+`|byte-range:${t}`}async stat(t,r){let{length:n}=n1(t);return{totalSize:n}}async read(t,r){let n=n1(t);return new tt(this.base,n).read(r)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}get singleKey(){return!0}};function Ik(){return{scheme:"byte-range",description:"byte range slicing",getKvStore(e,t){return Rr(e),{store:new sc(new Te(t.store,t.path)),path:e.suffix??""}}}}Wt.registerKvStoreAdapterProvider(Ik);var i1=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),s1=e=>{throw TypeError(e)},_k=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&s1("Object expected");var n,i;r&&(n=t[i1("asyncDispose")]),n===void 0&&(n=t[i1("dispose")],r&&(i=n)),typeof n!="function"&&s1("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},Ck=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,u){return u=Error(c),u.name="SuppressedError",u.error=o,u.suppressed=a,u},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};var Si=class{constructor(t,r=`gs://${t}/`,n=Re){this.bucket=t,this.baseUrlForDisplay=r,this.fetchOkImpl=n}getObjectUrl(t){return`https://storage.googleapis.com/storage/v1/b/${this.bucket}/o/${encodeURIComponent(t)}?alt=media&neuroglancer=${vl()}`}stat(t,r){return pn(this,t,this.getObjectUrl(t),r,this.fetchOkImpl)}read(t,r){return ei(this,t,this.getObjectUrl(t),r,this.fetchOkImpl)}async list(t,r){var n=[];try{let{progressListener:o}=r,a=_k(n,o===void 0?void 0:new He(o,{message:`Listing prefix ${this.getUrl(t)}`})),l=await(await this.fetchOkImpl(`https://storage.googleapis.com/storage/v1/b/${this.bucket}/o?delimiter=${encodeURIComponent("/")}&prefix=${encodeURIComponent(t)}&neuroglancerOrigin=${encodeURIComponent(location.origin)}`,{signal:r.signal,progressListener:r.progressListener})).json();Ee(l);let f=nr(l,"prefixes",Yt,[]).map(p=>p.substring(0,p.length-1)),h=nr(l,"items",p=>ct(p,d=>(Ee(d),re(d,"name",et))),[]).filter(p=>!p.endsWith("_$folder$")).map(p=>({key:p}));return{directories:f,entries:h}}catch(o){var i=o,s=!0}finally{Ck(n,i,s)}}getUrl(t){return this.baseUrlForDisplay+Ze(t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function Ak(e){return{scheme:"gs",description:Ss?"Google Cloud Storage":"Google Cloud Storage (anonymous)",getKvStore(t){let r=(t.suffix??"").match(/^\/\/([^/]+)(\/.*)?$/);if(r===null)throw new Error("Invalid URL, expected `gs://<bucket>/<path>`");let[,n,i]=r;return{store:new Si(n),path:decodeURIComponent((i??"").substring(1))}}}}Wt.registerBaseKvStoreProvider(Ak);var oc=class{constructor(t,r,n){this.base=t,this.scheme=r,this.format=n}getUrl(t){return this.validatePath(t),this.base.getUrl()+`|${this.scheme}`}validatePath(t){if(t)throw new Error(`"${this.scheme}:" does not support non-empty path ${JSON.stringify(t)}`)}async stat(t,r){return this.validatePath(t),await this.base.stat(r),{totalSize:void 0}}async read(t,r){return this.validatePath(t),new jr(this.base,this.format).read(r)}get supportsOffsetReads(){return!1}get supportsSuffixReads(){return!1}get singleKey(){return!0}};async function Mk(e){return _0(e.prefix)?[{suffix:"gzip:",description:"gzip-compressed"}]:[]}function o1(e){e.registerFileFormat({prefixLength:3,suffixLength:0,match:Mk})}function Tk(e,t){return{scheme:e,description:`transparent ${e} decoding`,getKvStore(r,n){return _g(r),{store:new oc(new Te(n.store,n.path),e,t),path:""}}}}Wt.registerKvStoreAdapterProvider(()=>Tk("gzip","gzip"));o1(Wt.autoDetectRegistry);xt(kg,async function(e,t){return{value:await this.get(e.sharedKvStoreContext).kvStoreContext.stat(e.url,t)}});xt(Ng,async function(e,t){let n=await this.get(e.sharedKvStoreContext).kvStoreContext.read(e.url,{...t,byteRange:e.byteRange,throwIfMissing:e.throwIfMissing});if(n===void 0)return{value:void 0};let i=await n.response.arrayBuffer();return{value:{data:i,offset:n.offset,totalSize:n.totalSize},transfers:[i]}});function ac(e,t,r){return e.rpc.promiseInvoke(tl,{sharedKvStoreContext:e.rpcId,url:t},{signal:r.signal,progressListener:r.progressListener})}xt(tl,async function(e,t){let r=this.get(e.sharedKvStoreContext),{store:n,path:i}=r.kvStoreContext.getKvStore(e.url);return{value:await n.list(i,t)}});xt(Dg,async function(e,t){let r=this.get(e.sharedKvStoreContext),{kvStoreContext:n}=r,{url:i}=e,s=bg(i),o;if(s===i){let a=bo(s),c=n.getBaseKvStoreProvider(a);c.completeUrl!==void 0&&(o=await c.completeUrl({url:a,...t}))}else{let a=bo(s),c=n.getKvStoreAdapterProvider(a),u=i.slice(0,i.length-s.length-1),l=n.getKvStore(u);c.completeUrl!==void 0&&(o=await c.completeUrl({url:a,base:l,...t}))}return{value:o}});var wi=class extends ti{list(t,r){return ac(this.sharedKvStoreContext,this.getUrl(t),r)}};fv(wt,wi);function a1(e,t){let{nodes:r}=e,n=we(0,r.length,a=>r[a].path>=t),i=we(Math.min(r.length,n+1),r.length,a=>!r[a].path.startsWith(t)),s={entries:[],directories:[]};for(let a=n;a<i;){let c=r[a],{path:u}=c,l=u.indexOf("/",t.length);if(l===-1)++a;else{l+1===u.length&&s.directories.push(u.slice(0,l));let f=u.substring(0,l+1);a=we(a+1,i,h=>!r[h].path.startsWith(f))}}let o=t.lastIndexOf("/");if("zarr.json".startsWith(t.slice(o+1))){let a=t.substring(0,o+1);if(er(r,a,(u,l)=>St(u,l.path))>=0)s.entries.push({key:a+"zarr.json"});else throw new Error(`Parent node ${JSON.stringify(a)} not found`)}return Eo(s)}var cc,kk={lang:void 0,message:void 0,abortEarly:void 0,abortPipeEarly:void 0};function l1(e){return!e&&!cc?kk:{lang:e?.lang??cc?.lang,message:e?.message,abortEarly:e?.abortEarly??cc?.abortEarly,abortPipeEarly:e?.abortPipeEarly??cc?.abortPipeEarly}}var Nk;function Dk(e){return Nk?.get(e)}var Pk;function Rk(e){return Pk?.get(e)}var Ok;function Uk(e,t){return Ok?.get(e)?.get(t)}function f1(e){let t=typeof e;return t==="string"?`"${e}"`:t==="number"||t==="bigint"||t==="boolean"?`${e}`:t==="object"||t==="function"?(e&&Object.getPrototypeOf(e)?.constructor?.name)??"null":t}function Pe(e,t,r,n,i){let s=i&&"input"in i?i.input:r.value,o=i?.expected??e.expects??null,a=i?.received??f1(s),c={kind:e.kind,type:e.type,input:s,expected:o,received:a,message:`Invalid ${t}: ${o?`Expected ${o} but r`:"R"}eceived ${a}`,requirement:e.requirement,path:i?.path,issues:i?.issues,lang:n.lang,abortEarly:n.abortEarly,abortPipeEarly:n.abortPipeEarly},u=e.kind==="schema",l=i?.message??e.message??Uk(e.reference,c.lang)??(u?Rk(c.lang):null)??n.message??Dk(c.lang);l!==void 0&&(c.message=typeof l=="function"?l(c):l),u&&(r.typed=!1),r.issues?r.issues.push(c):r.issues=[c]}var c1=new WeakMap;function st(e){let t=c1.get(e);return t||(t={version:1,vendor:"valibot",validate(r){return e["~run"]({value:r},l1())}},c1.set(e,t)),t}function Lk(e,t){return Object.prototype.hasOwnProperty.call(e,t)&&t!=="__proto__"&&t!=="prototype"&&t!=="constructor"}function h1(e,t){let r=[...new Set(e)];return r.length>1?`(${r.join(` ${t} `)})`:r[0]??"never"}function Fk(e){if(e.path){let t="";for(let r of e.path)if(typeof r.key=="string"||typeof r.key=="number")t?t+=`.${r.key}`:t+=r.key;else return null;return t}return null}function p1(e){return e instanceof d1}var d1=class extends Error{constructor(e){super(e[0].message),this.name="ValiError",this.issues=e}};function eh(e,t){return{kind:"validation",type:"check",reference:eh,async:!1,expects:null,requirement:e,message:t,"~run"(r,n){return r.typed&&!this.requirement(r.value)&&Pe(this,"input",r,n),r}}}function th(e){return{kind:"validation",type:"integer",reference:th,async:!1,expects:null,requirement:Number.isInteger,message:e,"~run"(t,r){return t.typed&&!this.requirement(t.value)&&Pe(this,"integer",t,r),t}}}function bs(e,t){return{kind:"validation",type:"length",reference:bs,async:!1,expects:`${e}`,requirement:e,message:t,"~run"(r,n){return r.typed&&r.value.length!==this.requirement&&Pe(this,"length",r,n,{received:`${r.value.length}`}),r}}}function Ye(e){return{kind:"transformation",type:"transform",reference:Ye,async:!1,operation:e,"~run"(t){return t.value=this.operation(t.value),t}}}function Bk(e,t,r){return typeof e.fallback=="function"?e.fallback(t,r):e.fallback}function m1(e){let t={};for(let r of e)if(r.path){let n=Fk(r);n?(t.nested||(t.nested={}),Object.prototype.hasOwnProperty.call(t.nested,n)?t.nested[n].push(r.message):t.nested[n]=[r.message]):t.other?t.other.push(r.message):t.other=[r.message]}else t.root?t.root.push(r.message):t.root=[r.message];return t}function g1(e,t,r){return typeof e.default=="function"?e.default(t,r):e.default}function _t(){return{kind:"schema",type:"any",reference:_t,expects:"any",async:!1,get"~standard"(){return st(this)},"~run"(e){return e.typed=!0,e}}}function ot(e,t){return{kind:"schema",type:"array",reference:ot,expects:"Array",async:!1,item:e,message:t,get"~standard"(){return st(this)},"~run"(r,n){let i=r.value;if(Array.isArray(i)){r.typed=!0,r.value=[];for(let s=0;s<i.length;s++){let o=i[s],a=this.item["~run"]({value:o},n);if(a.issues){let c={type:"array",origin:"value",input:i,key:s,value:o};for(let u of a.issues)u.path?u.path.unshift(c):u.path=[c],r.issues?.push(u);if(r.issues||(r.issues=a.issues),n.abortEarly){r.typed=!1;break}}a.typed||(r.typed=!1),r.value.push(a.value)}}else Pe(this,"type",r,n);return r}}}function rh(e){return{kind:"schema",type:"bigint",reference:rh,expects:"bigint",async:!1,message:e,get"~standard"(){return st(this)},"~run"(t,r){return typeof t.value=="bigint"?t.typed=!0:Pe(this,"type",t,r),t}}}function Is(e,t){return{kind:"schema",type:"instance",reference:Is,expects:e.name,async:!1,class:e,message:t,get"~standard"(){return st(this)},"~run"(r,n){return r.value instanceof this.class?r.typed=!0:Pe(this,"type",r,n),r}}}function Xt(e,t,r){return{kind:"schema",type:"map",reference:Xt,expects:"Map",async:!1,key:e,value:t,message:r,get"~standard"(){return st(this)},"~run"(n,i){let s=n.value;if(s instanceof Map){n.typed=!0,n.value=new Map;for(let[o,a]of s){let c=this.key["~run"]({value:o},i);if(c.issues){let l={type:"map",origin:"key",input:s,key:o,value:a};for(let f of c.issues)f.path?f.path.unshift(l):f.path=[l],n.issues?.push(f);if(n.issues||(n.issues=c.issues),i.abortEarly){n.typed=!1;break}}let u=this.value["~run"]({value:a},i);if(u.issues){let l={type:"map",origin:"value",input:s,key:o,value:a};for(let f of u.issues)f.path?f.path.unshift(l):f.path=[l],n.issues?.push(f);if(n.issues||(n.issues=u.issues),i.abortEarly){n.typed=!1;break}}(!c.typed||!u.typed)&&(n.typed=!1),n.value.set(c.value,u.value)}}else Pe(this,"type",n,i);return n}}}function _s(e,t){return{kind:"schema",type:"nullable",reference:_s,expects:`(${e.expects} | null)`,async:!1,wrapped:e,default:t,get"~standard"(){return st(this)},"~run"(r,n){return r.value===null&&(this.default!==void 0&&(r.value=g1(this,r,n)),r.value===null)?(r.typed=!0,r):this.wrapped["~run"](r,n)}}}function nh(e){return{kind:"schema",type:"number",reference:nh,expects:"number",async:!1,message:e,get"~standard"(){return st(this)},"~run"(t,r){return typeof t.value=="number"&&!isNaN(t.value)?t.typed=!0:Pe(this,"type",t,r),t}}}function uc(e,t){return{kind:"schema",type:"picklist",reference:uc,expects:h1(e.map(f1),"|"),async:!1,options:e,message:t,get"~standard"(){return st(this)},"~run"(r,n){return this.options.includes(r.value)?r.typed=!0:Pe(this,"type",r,n),r}}}function ih(e,t,r){return{kind:"schema",type:"record",reference:ih,expects:"Object",async:!1,key:e,value:t,message:r,get"~standard"(){return st(this)},"~run"(n,i){let s=n.value;if(s&&typeof s=="object"){n.typed=!0,n.value={};for(let o in s)if(Lk(s,o)){let a=s[o],c=this.key["~run"]({value:o},i);if(c.issues){let l={type:"object",origin:"key",input:s,key:o,value:a};for(let f of c.issues)f.path=[l],n.issues?.push(f);if(n.issues||(n.issues=c.issues),i.abortEarly){n.typed=!1;break}}let u=this.value["~run"]({value:a},i);if(u.issues){let l={type:"object",origin:"value",input:s,key:o,value:a};for(let f of u.issues)f.path?f.path.unshift(l):f.path=[l],n.issues?.push(f);if(n.issues||(n.issues=u.issues),i.abortEarly){n.typed=!1;break}}(!c.typed||!u.typed)&&(n.typed=!1),c.typed&&(n.value[c.value]=u.value)}}else Pe(this,"type",n,i);return n}}}function lr(e,t){return{kind:"schema",type:"strict_object",reference:lr,expects:"Object",async:!1,entries:e,message:t,get"~standard"(){return st(this)},"~run"(r,n){let i=r.value;if(i&&typeof i=="object"){r.typed=!0,r.value={};for(let s in this.entries){let o=this.entries[s];if(s in i||(o.type==="exact_optional"||o.type==="optional"||o.type==="nullish")&&o.default!==void 0){let a=s in i?i[s]:g1(o),c=o["~run"]({value:a},n);if(c.issues){let u={type:"object",origin:"value",input:i,key:s,value:a};for(let l of c.issues)l.path?l.path.unshift(u):l.path=[u],r.issues?.push(l);if(r.issues||(r.issues=c.issues),n.abortEarly){r.typed=!1;break}}c.typed||(r.typed=!1),r.value[s]=c.value}else if(o.fallback!==void 0)r.value[s]=Bk(o);else if(o.type!=="exact_optional"&&o.type!=="optional"&&o.type!=="nullish"&&(Pe(this,"key",r,n,{input:void 0,expected:`"${s}"`,path:[{type:"object",origin:"key",input:i,key:s,value:i[s]}]}),n.abortEarly))break}if(!r.issues||!n.abortEarly){for(let s in i)if(!(s in this.entries)){Pe(this,"key",r,n,{input:s,expected:"never",path:[{type:"object",origin:"key",input:i,key:s,value:i[s]}]});break}}}else Pe(this,"type",r,n);return r}}}function sh(e,t){return{kind:"schema",type:"strict_tuple",reference:sh,expects:"Array",async:!1,items:e,message:t,get"~standard"(){return st(this)},"~run"(r,n){let i=r.value;if(Array.isArray(i)){r.typed=!0,r.value=[];for(let s=0;s<this.items.length;s++){let o=i[s],a=this.items[s]["~run"]({value:o},n);if(a.issues){let c={type:"array",origin:"value",input:i,key:s,value:o};for(let u of a.issues)u.path?u.path.unshift(c):u.path=[c],r.issues?.push(u);if(r.issues||(r.issues=a.issues),n.abortEarly){r.typed=!1;break}}a.typed||(r.typed=!1),r.value.push(a.value)}!(r.issues&&n.abortEarly)&&this.items.length<i.length&&Pe(this,"type",r,n,{input:i[this.items.length],expected:"never",path:[{type:"array",origin:"value",input:i,key:this.items.length,value:i[this.items.length]}]})}else Pe(this,"type",r,n);return r}}}function Ue(e){return{kind:"schema",type:"string",reference:Ue,expects:"string",async:!1,message:e,get"~standard"(){return st(this)},"~run"(t,r){return typeof t.value=="string"?t.typed=!0:Pe(this,"type",t,r),t}}}function oh(e,t){return{kind:"schema",type:"tuple",reference:oh,expects:"Array",async:!1,items:e,message:t,get"~standard"(){return st(this)},"~run"(r,n){let i=r.value;if(Array.isArray(i)){r.typed=!0,r.value=[];for(let s=0;s<this.items.length;s++){let o=i[s],a=this.items[s]["~run"]({value:o},n);if(a.issues){let c={type:"array",origin:"value",input:i,key:s,value:o};for(let u of a.issues)u.path?u.path.unshift(c):u.path=[c],r.issues?.push(u);if(r.issues||(r.issues=a.issues),n.abortEarly){r.typed=!1;break}}a.typed||(r.typed=!1),r.value.push(a.value)}}else Pe(this,"type",r,n);return r}}}function u1(e){let t;if(e)for(let r of e)if(t)for(let n of r.issues)t.push(n);else t=r.issues;return t}function Hr(e,t){return{kind:"schema",type:"union",reference:Hr,expects:h1(e.map(r=>r.expects),"|"),async:!1,options:e,message:t,get"~standard"(){return st(this)},"~run"(r,n){let i,s,o;for(let a of this.options){let c=a["~run"]({value:r.value},n);if(c.typed)if(c.issues)s?s.push(c):s=[c];else{i=c;break}else o?o.push(c):o=[c]}if(i)return i;if(s){if(s.length===1)return s[0];Pe(this,"type",r,n,{issues:u1(s)}),r.typed=!0}else{if(o?.length===1)return o[0];Pe(this,"type",r,n,{issues:u1(o)})}return r}}}function y1(e,t,r){let n=e["~run"]({value:t},l1(r));if(n.issues)throw new d1(n.issues);return n.value}function Je(...e){return{...e[0],pipe:e,get"~standard"(){return st(this)},"~run"(t,r){for(let n of e)if(n.kind!=="metadata"){if(t.issues&&(n.kind==="schema"||n.kind==="transformation")){t.typed=!1;break}(!t.issues||!r.abortEarly&&!r.abortPipeEarly)&&(t=n["~run"](t,r))}return t}}}var uh;try{uh=new TextDecoder}catch{}var $,zt,A=0;var N1=[],lh=N1,fh=0,ve={},ue,Wr,Bt=0,fr=0,Fe,Cr,dt=[],ce,v1={useRecords:!1,mapsAsObjects:!0},Cs=class{},ph=new Cs;ph.name="MessagePack 0xC1";var Xr=!1,x1=2,S1,w1,E1;var Ar=class e{constructor(t){t&&(t.useRecords===!1&&t.mapsAsObjects===void 0&&(t.mapsAsObjects=!0),t.sequential&&t.trusted!==!1&&(t.trusted=!0,!t.structures&&t.useRecords!=!1&&(t.structures=[],t.maxSharedStructures||(t.maxSharedStructures=0))),t.structures?t.structures.sharedLength=t.structures.length:t.getStructures&&((t.structures=[]).uninitialized=!0,t.structures.sharedLength=0),t.int64AsNumber&&(t.int64AsType="number")),Object.assign(this,t)}unpack(t,r){if($)return U1(()=>(fc(),this?this.unpack(t,r):e.prototype.unpack.call(v1,t,r)));!t.buffer&&t.constructor===ArrayBuffer&&(t=typeof Buffer<"u"?Buffer.from(t):new Uint8Array(t)),typeof r=="object"?(zt=r.end||t.length,A=r.start||0):(A=0,zt=r>-1?r:t.length),fh=0,fr=0,Wr=null,lh=N1,Fe=null,$=t;try{ce=t.dataView||(t.dataView=new DataView(t.buffer,t.byteOffset,t.byteLength))}catch(n){throw $=null,t instanceof Uint8Array?n:new Error("Source must be a Uint8Array or Buffer but was a "+(t&&typeof t=="object"?t.constructor.name:typeof t))}if(this instanceof e){if(ve=this,this.structures)return ue=this.structures,lc(r);(!ue||ue.length>0)&&(ue=[])}else ve=v1,(!ue||ue.length>0)&&(ue=[]);return lc(r)}unpackMultiple(t,r){let n,i=0;try{Xr=!0;let s=t.length,o=this?this.unpack(t,s):pc.unpack(t,s);if(r){if(r(o,i,A)===!1)return;for(;A<s;)if(i=A,r(lc(),i,A)===!1)return}else{for(n=[o];A<s;)i=A,n.push(lc());return n}}catch(s){throw s.lastPosition=i,s.values=n,s}finally{Xr=!1,fc()}}_mergeStructures(t,r){w1&&(t=w1.call(this,t)),t=t||[],Object.isFrozen(t)&&(t=t.map(n=>n.slice(0)));for(let n=0,i=t.length;n<i;n++){let s=t[n];s&&(s.isShared=!0,n>=32&&(s.highByte=n-32>>5))}t.sharedLength=t.length;for(let n in r||[])if(n>=0){let i=t[n],s=r[n];s&&(i&&((t.restoreStructures||(t.restoreStructures=[]))[n]=i),t[n]=s)}return this.structures=t}decode(t,r){return this.unpack(t,r)}};function lc(e){try{if(!ve.trusted&&!Xr){let r=ue.sharedLength||0;r<ue.length&&(ue.length=r)}let t;if(ve.randomAccessStructure&&$[A]<64&&$[A]>=32&&S1?(t=S1($,A,zt,ve),$=null,!(e&&e.lazy)&&t&&(t=t.toJSON()),A=zt):t=Me(),Fe&&(A=Fe.postBundlePosition,Fe=null),Xr&&(ue.restoreStructures=null),A==zt)ue&&ue.restoreStructures&&b1(),ue=null,$=null,Cr&&(Cr=null);else{if(A>zt)throw new Error("Unexpected end of MessagePack data");if(!Xr){let r;try{r=JSON.stringify(t,(n,i)=>typeof i=="bigint"?`${i}n`:i).slice(0,100)}catch(n){r="(JSON view not available "+n+")"}throw new Error("Data read, but end of buffer not reached "+r)}}return t}catch(t){throw ue&&ue.restoreStructures&&b1(),fc(),(t instanceof RangeError||t.message.startsWith("Unexpected end of buffer")||A>zt)&&(t.incomplete=!0),t}}function b1(){for(let e in ue.restoreStructures)ue[e]=ue.restoreStructures[e];ue.restoreStructures=null}function Me(){let e=$[A++];if(e<160)if(e<128){if(e<64)return e;{let t=ue[e&63]||ve.getStructures&&D1()[e&63];return t?(t.read||(t.read=dh(t,e&63)),t.read()):e}}else if(e<144)if(e-=128,ve.mapsAsObjects){let t={};for(let r=0;r<e;r++){let n=R1();n==="__proto__"&&(n="__proto_"),t[n]=Me()}return t}else{let t=new Map;for(let r=0;r<e;r++)t.set(Me(),Me());return t}else{e-=144;let t=new Array(e);for(let r=0;r<e;r++)t[r]=Me();return ve.freezeData?Object.freeze(t):t}else if(e<192){let t=e-160;if(fr>=A)return Wr.slice(A-Bt,(A+=t)-Bt);if(fr==0&&zt<140){let r=t<16?mh(t):P1(t);if(r!=null)return r}return hh(t)}else{let t;switch(e){case 192:return null;case 193:return Fe?(t=Me(),t>0?Fe[1].slice(Fe.position1,Fe.position1+=t):Fe[0].slice(Fe.position0,Fe.position0-=t)):ph;case 194:return!1;case 195:return!0;case 196:if(t=$[A++],t===void 0)throw new Error("Unexpected end of buffer");return ch(t);case 197:return t=ce.getUint16(A),A+=2,ch(t);case 198:return t=ce.getUint32(A),A+=4,ch(t);case 199:return vn($[A++]);case 200:return t=ce.getUint16(A),A+=2,vn(t);case 201:return t=ce.getUint32(A),A+=4,vn(t);case 202:if(t=ce.getFloat32(A),ve.useFloat32>2){let r=hc[($[A]&127)<<1|$[A+1]>>7];return A+=4,(r*t+(t>0?.5:-.5)>>0)/r}return A+=4,t;case 203:return t=ce.getFloat64(A),A+=8,t;case 204:return $[A++];case 205:return t=ce.getUint16(A),A+=2,t;case 206:return t=ce.getUint32(A),A+=4,t;case 207:return ve.int64AsType==="number"?(t=ce.getUint32(A)*4294967296,t+=ce.getUint32(A+4)):ve.int64AsType==="string"?t=ce.getBigUint64(A).toString():ve.int64AsType==="auto"?(t=ce.getBigUint64(A),t<=BigInt(2)<<BigInt(52)&&(t=Number(t))):t=ce.getBigUint64(A),A+=8,t;case 208:return ce.getInt8(A++);case 209:return t=ce.getInt16(A),A+=2,t;case 210:return t=ce.getInt32(A),A+=4,t;case 211:return ve.int64AsType==="number"?(t=ce.getInt32(A)*4294967296,t+=ce.getUint32(A+4)):ve.int64AsType==="string"?t=ce.getBigInt64(A).toString():ve.int64AsType==="auto"?(t=ce.getBigInt64(A),t>=BigInt(-2)<<BigInt(52)&&t<=BigInt(2)<<BigInt(52)&&(t=Number(t))):t=ce.getBigInt64(A),A+=8,t;case 212:if(t=$[A++],t==114)return T1($[A++]&63);{let r=dt[t];if(r)return r.read?(A++,r.read(Me())):r.noBuffer?(A++,r()):r($.subarray(A,++A));throw new Error("Unknown extension "+t)}case 213:return t=$[A],t==114?(A++,T1($[A++]&63,$[A++])):vn(2);case 214:return vn(4);case 215:return vn(8);case 216:return vn(16);case 217:return t=$[A++],fr>=A?Wr.slice(A-Bt,(A+=t)-Bt):$k(t);case 218:return t=ce.getUint16(A),A+=2,fr>=A?Wr.slice(A-Bt,(A+=t)-Bt):Vk(t);case 219:return t=ce.getUint32(A),A+=4,fr>=A?Wr.slice(A-Bt,(A+=t)-Bt):Gk(t);case 220:return t=ce.getUint16(A),A+=2,_1(t);case 221:return t=ce.getUint32(A),A+=4,_1(t);case 222:return t=ce.getUint16(A),A+=2,C1(t);case 223:return t=ce.getUint32(A),A+=4,C1(t);default:if(e>=224)return e-256;if(e===void 0){let r=new Error("Unexpected end of MessagePack data");throw r.incomplete=!0,r}throw new Error("Unknown MessagePack token "+e)}}}var zk=/^[a-zA-Z_$][a-zA-Z\d_$]*$/;function dh(e,t){function r(){if(r.count++>x1){let i;try{i=e.read=new Function("r","return function(){return "+(ve.freezeData?"Object.freeze":"")+"({"+e.map(s=>s==="__proto__"?"__proto_:r()":zk.test(s)?s+":r()":"["+JSON.stringify(s)+"]:r()").join(",")+"})}")(Me)}catch{return x1=1/0,r()}return e.read0=i,e.highByte===0&&(e.read=I1(t,e.read)),i()}let n={};for(let i=0,s=e.length;i<s;i++){let o=e[i];o==="__proto__"&&(o="__proto_"),n[o]=Me()}return ve.freezeData?Object.freeze(n):n}return r.count=0,e.read0=r,e.highByte===0?I1(t,r):r}var I1=(e,t)=>function(){let r=$[A++];if(r===0)return t();let n=e<32?-(e+(r<<5)):e+(r<<5),i=ue[n]||D1()[n];if(!i)throw new Error("Record id is not defined for "+n);return i.read||(i.read=dh(i,e)),i.read()};function D1(){let e=U1(()=>($=null,ve.getStructures()));return ue=ve._mergeStructures(e,ue)}var hh=As,$k=As,Vk=As,Gk=As;function As(e){let t;if(e<16&&(t=mh(e)))return t;if(e>64&&uh)return uh.decode($.subarray(A,A+=e));let r=A+e,n=[];for(t="";A<r;){let i=$[A++];if((i&128)===0)n.push(i);else if((i&224)===192){let s=$[A++]&63,o=(i&31)<<6|s;o<128?n.push(65533):n.push(o)}else if((i&240)===224){let s=$[A++]&63,o=$[A++]&63,a=(i&31)<<12|s<<6|o;a<2048||a>=55296&&a<=57343?n.push(65533):n.push(a)}else if((i&248)===240){let s=$[A++]&63,o=$[A++]&63,a=$[A++]&63,c=(i&7)<<18|s<<12|o<<6|a;c<65536||c>1114111?n.push(65533):(c>65535&&(c-=65536,n.push(c>>>10&1023|55296),c=56320|c&1023),n.push(c))}else n.push(65533);n.length>=4096&&(t+=Le.apply(String,n),n.length=0)}return n.length>0&&(t+=Le.apply(String,n)),t}function _1(e){let t=new Array(e);for(let r=0;r<e;r++)t[r]=Me();return ve.freezeData?Object.freeze(t):t}function C1(e){if(ve.mapsAsObjects){let t={};for(let r=0;r<e;r++){let n=R1();n==="__proto__"&&(n="__proto_"),t[n]=Me()}return t}else{let t=new Map;for(let r=0;r<e;r++)t.set(Me(),Me());return t}}var Le=String.fromCharCode;function P1(e){let t=A,r=new Array(e);for(let n=0;n<e;n++){let i=$[A++];if((i&128)>0){A=t;return}r[n]=i}return Le.apply(String,r)}function mh(e){if(e<4)if(e<2){if(e===0)return"";{let t=$[A++];if((t&128)>1){A-=1;return}return Le(t)}}else{let t=$[A++],r=$[A++];if((t&128)>0||(r&128)>0){A-=2;return}if(e<3)return Le(t,r);let n=$[A++];if((n&128)>0){A-=3;return}return Le(t,r,n)}else{let t=$[A++],r=$[A++],n=$[A++],i=$[A++];if((t&128)>0||(r&128)>0||(n&128)>0||(i&128)>0){A-=4;return}if(e<6){if(e===4)return Le(t,r,n,i);{let s=$[A++];if((s&128)>0){A-=5;return}return Le(t,r,n,i,s)}}else if(e<8){let s=$[A++],o=$[A++];if((s&128)>0||(o&128)>0){A-=6;return}if(e<7)return Le(t,r,n,i,s,o);let a=$[A++];if((a&128)>0){A-=7;return}return Le(t,r,n,i,s,o,a)}else{let s=$[A++],o=$[A++],a=$[A++],c=$[A++];if((s&128)>0||(o&128)>0||(a&128)>0||(c&128)>0){A-=8;return}if(e<10){if(e===8)return Le(t,r,n,i,s,o,a,c);{let u=$[A++];if((u&128)>0){A-=9;return}return Le(t,r,n,i,s,o,a,c,u)}}else if(e<12){let u=$[A++],l=$[A++];if((u&128)>0||(l&128)>0){A-=10;return}if(e<11)return Le(t,r,n,i,s,o,a,c,u,l);let f=$[A++];if((f&128)>0){A-=11;return}return Le(t,r,n,i,s,o,a,c,u,l,f)}else{let u=$[A++],l=$[A++],f=$[A++],h=$[A++];if((u&128)>0||(l&128)>0||(f&128)>0||(h&128)>0){A-=12;return}if(e<14){if(e===12)return Le(t,r,n,i,s,o,a,c,u,l,f,h);{let p=$[A++];if((p&128)>0){A-=13;return}return Le(t,r,n,i,s,o,a,c,u,l,f,h,p)}}else{let p=$[A++],d=$[A++];if((p&128)>0||(d&128)>0){A-=14;return}if(e<15)return Le(t,r,n,i,s,o,a,c,u,l,f,h,p,d);let m=$[A++];if((m&128)>0){A-=15;return}return Le(t,r,n,i,s,o,a,c,u,l,f,h,p,d,m)}}}}}function A1(){let e=$[A++],t;if(e<192)t=e-160;else switch(e){case 217:t=$[A++];break;case 218:t=ce.getUint16(A),A+=2;break;case 219:t=ce.getUint32(A),A+=4;break;default:throw new Error("Expected string")}return As(t)}function ch(e){return ve.copyBuffers?Uint8Array.prototype.slice.call($,A,A+=e):$.subarray(A,A+=e)}function vn(e){let t=$[A++];if(dt[t]){let r;return dt[t]($.subarray(A,r=A+=e),n=>{A=n;try{return Me()}finally{A=r}})}else throw new Error("Unknown extension type "+t)}var M1=new Array(4096);function R1(){let e=$[A++];if(e>=160&&e<192){if(e=e-160,fr>=A)return Wr.slice(A-Bt,(A+=e)-Bt);if(!(fr==0&&zt<180))return hh(e)}else return A--,O1(Me());let t=(e<<5^(e>1?ce.getUint16(A):e>0?$[A]:0))&4095,r=M1[t],n=A,i=A+e-3,s,o=0;if(r&&r.bytes==e){for(;n<i;){if(s=ce.getUint32(n),s!=r[o++]){n=1879048192;break}n+=4}for(i+=3;n<i;)if(s=$[n++],s!=r[o++]){n=1879048192;break}if(n===i)return A=n,r.string;i-=3,n=A}for(r=[],M1[t]=r,r.bytes=e;n<i;)s=ce.getUint32(n),r.push(s),n+=4;for(i+=3;n<i;)s=$[n++],r.push(s);let a=e<16?mh(e):P1(e);return a!=null?r.string=a:r.string=hh(e)}function O1(e){if(typeof e=="string")return e;if(typeof e=="number"||typeof e=="boolean"||typeof e=="bigint")return e.toString();if(e==null)return e+"";if(ve.allowArraysInMapKeys&&Array.isArray(e)&&e.flat().every(t=>["string","number","boolean","bigint"].includes(typeof t)))return e.flat().toString();throw new Error(`Invalid property type for record: ${typeof e}`)}var T1=(e,t)=>{let r=Me().map(O1),n=e;t!==void 0&&(e=e<32?-((t<<5)+e):(t<<5)+e,r.highByte=t);let i=ue[e];return i&&(i.isShared||Xr)&&((ue.restoreStructures||(ue.restoreStructures=[]))[e]=i),ue[e]=r,r.read=dh(r,n),(r.read0||r.read)()};dt[0]=()=>{};dt[0].noBuffer=!0;dt[66]=e=>{let t=e.byteLength%8||8,r=BigInt(e[0]&128?e[0]-256:e[0]);for(let n=1;n<t;n++)r<<=BigInt(8),r+=BigInt(e[n]);if(e.byteLength!==t){let n=new DataView(e.buffer,e.byteOffset,e.byteLength),i=(s,o)=>{let a=o-s;if(a<=40){let f=n.getBigUint64(s);for(let h=s+8;h<o;h+=8)f<<=BigInt(64),f|=n.getBigUint64(h);return f}let c=s+(a>>4<<3),u=i(s,c),l=i(c,o);return u<<BigInt((o-c)*8)|l};r=r<<BigInt((n.byteLength-t)*8)|i(t,n.byteLength)}return r};var k1={Error,EvalError,RangeError,ReferenceError,SyntaxError,TypeError,URIError,AggregateError:typeof AggregateError=="function"?AggregateError:null};dt[101]=()=>{let e=Me();if(!k1[e[0]]){let t=Error(e[1],{cause:e[2]});return t.name=e[0],t}return k1[e[0]](e[1],{cause:e[2]})};dt[105]=e=>{if(ve.structuredClone===!1)throw new Error("Structured clone extension is disabled");let t=ce.getUint32(A-4);Cr||(Cr=new Map);let r=$[A],n;r>=144&&r<160||r==220||r==221?n=[]:r>=128&&r<144||r==222||r==223?n=new Map:(r>=199&&r<=201||r>=212&&r<=216)&&$[A+1]===115?n=new Set:n={};let i={target:n};Cr.set(t,i);let s=Me();if(i.used)Object.assign(n,s);else return i.target=s;if(n instanceof Map)for(let[o,a]of s.entries())n.set(o,a);if(n instanceof Set)for(let o of Array.from(s))n.add(o);return n};dt[112]=e=>{if(ve.structuredClone===!1)throw new Error("Structured clone extension is disabled");let t=ce.getUint32(A-4),r=Cr.get(t);return r.used=!0,r.target};dt[115]=()=>new Set(Me());var gh=["Int8","Uint8","Uint8Clamped","Int16","Uint16","Int32","Uint32","Float32","Float64","BigInt64","BigUint64"].map(e=>e+"Array"),jk=typeof globalThis=="object"?globalThis:window;dt[116]=e=>{let t=e[0],r=Uint8Array.prototype.slice.call(e,1).buffer,n=gh[t];if(!n){if(t===16)return r;if(t===17)return new DataView(r);throw new Error("Could not find typed array for code "+t)}return new jk[n](r)};dt[120]=()=>{let e=Me();return new RegExp(e[0],e[1])};var qk=[];dt[98]=e=>{let t=(e[0]<<24)+(e[1]<<16)+(e[2]<<8)+e[3],r=A;return A+=t-e.length,Fe=qk,Fe=[A1(),A1()],Fe.position0=0,Fe.position1=0,Fe.postBundlePosition=A,A=r,Me()};dt[255]=e=>e.length==4?new Date((e[0]*16777216+(e[1]<<16)+(e[2]<<8)+e[3])*1e3):e.length==8?new Date(((e[0]<<22)+(e[1]<<14)+(e[2]<<6)+(e[3]>>2))/1e6+((e[3]&3)*4294967296+e[4]*16777216+(e[5]<<16)+(e[6]<<8)+e[7])*1e3):e.length==12?new Date(((e[0]<<24)+(e[1]<<16)+(e[2]<<8)+e[3])/1e6+((e[4]&128?-281474976710656:0)+e[6]*1099511627776+e[7]*4294967296+e[8]*16777216+(e[9]<<16)+(e[10]<<8)+e[11])*1e3):new Date("invalid");function U1(e){E1&&E1();let t=zt,r=A,n=fh,i=Bt,s=fr,o=Wr,a=lh,c=Cr,u=Fe,l=new Uint8Array($.slice(0,zt)),f=ue,h=ue.slice(0,ue.length),p=ve,d=Xr,m=e();return zt=t,A=r,fh=n,Bt=i,fr=s,Wr=o,lh=a,Cr=c,Fe=u,$=l,Xr=d,ue=f,ue.splice(0,ue.length,...h),ve=p,ce=new DataView($.buffer,$.byteOffset,$.byteLength),m}function fc(){$=null,Cr=null,ue=null}var hc=new Array(147);for(let e=0;e<256;e++)hc[e]=+("1e"+Math.floor(45.15-e*.30103));var pc=new Ar({useRecords:!1}),Kk=pc.unpack,Yk=pc.unpackMultiple,Jk=pc.unpack,dc={NEVER:0,ALWAYS:1,DECIMAL_ROUND:3,DECIMAL_FIT:4},Hk=new Float32Array(1),Kj=new Uint8Array(Hk.buffer,0,4);var gc;try{gc=new TextEncoder}catch{}var xh,Sh,bi=typeof Buffer<"u",mc=bi?function(e){return Buffer.allocUnsafeSlow(e)}:Uint8Array,z1=bi?Buffer:Uint8Array,L1=bi?4294967296:2144337920,D,Ms,xe,k=0,Qe,_e=null,Wk,Xk=21760,Zk=/[\u0080-\uFFFF]/,Ei=Symbol("record-id"),Ts=class extends Ar{constructor(t){super(t),this.offset=0;let r,n,i,s,o,a=z1.prototype.utf8Write?function(S,O){return D.utf8Write(S,O,D.byteLength-O)}:gc&&gc.encodeInto?function(S,O){return gc.encodeInto(S,D.subarray(O)).written}:!1,c=this;t||(t={});let u=t&&t.sequential,l=t.structures||t.saveStructures,f=t.maxSharedStructures;if(f==null&&(f=l?32:0),f>8160)throw new Error("Maximum maxSharedStructure is 8160");t.structuredClone&&t.moreTypes==null&&(this.moreTypes=!0);let h=t.maxOwnStructures;h==null&&(h=l?32:64),!this.structures&&t.useRecords!=!1&&(this.structures=[]);let p=f>32||h+f>64,d=f+64,m=f+h+64;if(m>8256)throw new Error("Maximum maxSharedStructure + maxOwnStructure is 8192");let g=[],y=0,I=0;this.pack=this.encode=function(S,O){if(D||(D=new mc(8192),xe=D.dataView||(D.dataView=new DataView(D.buffer,0,8192)),k=0),Qe=D.length-10,Qe-k<2048?(D=new mc(D.length),xe=D.dataView||(D.dataView=new DataView(D.buffer,0,D.length)),Qe=D.length-10,k=0):k=k+7&2147483640,n=k,O&j1&&(k+=O&255),o=c.structuredClone?new Map:null,c.bundleStrings&&typeof S!="string"?(_e=[],_e.size=1/0):_e=null,s=c.structures,s){s.uninitialized&&(s=c._mergeStructures(c.getStructures()));let N=s.sharedLength||0;if(N>f)throw new Error("Shared structures is larger than maximum shared structures, try increasing maxSharedStructures to "+s.sharedLength);if(!s.transitions){s.transitions=Object.create(null);for(let U=0;U<N;U++){let L=s[U];if(!L)continue;let j,V=s.transitions;for(let Y=0,J=L.length;Y<J;Y++){let de=L[Y];j=V[de],j||(j=V[de]=Object.create(null)),V=j}V[Ei]=U+64}this.lastNamedStructuresLength=N}u||(s.nextId=N+64)}i&&(i=!1);let R;try{c.randomAccessStructure&&!c.readOnlyStructures&&S&&typeof S=="object"?S.constructor===Object?F(S):S.constructor!==Map&&!Array.isArray(S)&&!Sh.some(U=>S instanceof U)?F(S.toJSON?S.toJSON():S):b(S):b(S);let N=_e;if(_e&&B1(n,b,0),o&&o.idsToInsert){let U=o.idsToInsert.sort((Y,J)=>Y.offset>J.offset?1:-1),L=U.length,j=-1;for(;N&&L>0;){let Y=U[--L].offset+n;Y<N.stringsPosition+n&&j===-1&&(j=0),Y>N.position+n?j>=0&&(j+=6):(j>=0&&(xe.setUint32(N.position+n,xe.getUint32(N.position+n)+j),j=-1),N=N.previous,L++)}j>=0&&N&&xe.setUint32(N.position+n,xe.getUint32(N.position+n)+j),k+=U.length*6,k>Qe&&T(k),c.offset=k;let V=Qk(D.subarray(n,k),U);return o=null,V}return c.offset=k,O&V1?(D.start=n,D.end=k,D):D.subarray(n,k)}catch(N){throw R=N,N}finally{if(s&&(_(),i&&c.saveStructures)){let N=s.sharedLength||0,U=D.subarray(n,k),L=eN(s,c);if(!R)return c.saveStructures(L,L.isCompatible)===!1?(s.uninitialized=!0,c.pack(S,O)):(c.lastNamedStructuresLength=N,D.length>1073741824&&(D=null),U)}D.length>1073741824&&(D=null),O&G1&&(k=n)}};let _=()=>{I<10&&I++;let S=s.sharedLength||0;if(s.length>S&&!u&&(s.length=S),y>1e4)s.transitions=null,I=0,y=0,g.length>0&&(g=[]);else if(g.length>0&&!u){for(let O=0,R=g.length;O<R;O++)g[O][Ei]=0;g=[]}},E=S=>{var O=S.length;O<16?D[k++]=144|O:O<65536?(D[k++]=220,D[k++]=O>>8,D[k++]=O&255):(D[k++]=221,xe.setUint32(k,O),k+=4);for(let R=0;R<O;R++)b(S[R])},b=S=>{k>Qe&&(D=T(k));var O=typeof S,R;if(O==="string"){let N=S.length;if(_e&&N>=4&&N<4096){if((_e.size+=N)>Xk){let V,Y=(_e[0]?_e[0].length*3+_e[1].length:0)+10;k+Y>Qe&&(D=T(k+Y));let J;_e.position?(J=_e,D[k]=200,k+=3,D[k++]=98,V=k-n,k+=4,B1(n,b,0),xe.setUint16(V+n-3,k-n-V)):(D[k++]=214,D[k++]=98,V=k-n,k+=4),_e=["",""],_e.previous=J,_e.size=0,_e.position=V}let j=Zk.test(S);_e[j?0:1]+=S,D[k++]=193,b(j?-N:N);return}let U;N<32?U=1:N<256?U=2:N<65536?U=3:U=5;let L=N*3;if(k+L>Qe&&(D=T(k+L)),N<64||!a){let j,V,Y,J=k+U;for(j=0;j<N;j++)V=S.charCodeAt(j),V<128?D[J++]=V:V<2048?(D[J++]=V>>6|192,D[J++]=V&63|128):(V&64512)===55296&&((Y=S.charCodeAt(j+1))&64512)===56320?(V=65536+((V&1023)<<10)+(Y&1023),j++,D[J++]=V>>18|240,D[J++]=V>>12&63|128,D[J++]=V>>6&63|128,D[J++]=V&63|128):(D[J++]=V>>12|224,D[J++]=V>>6&63|128,D[J++]=V&63|128);R=J-k-U}else R=a(S,k+U);R<32?D[k++]=160|R:R<256?(U<2&&D.copyWithin(k+2,k+1,k+1+R),D[k++]=217,D[k++]=R):R<65536?(U<3&&D.copyWithin(k+3,k+2,k+2+R),D[k++]=218,D[k++]=R>>8,D[k++]=R&255):(U<5&&D.copyWithin(k+5,k+3,k+3+R),D[k++]=219,xe.setUint32(k,R),k+=4),k+=R}else if(O==="number")if(S>>>0===S)S<32||S<128&&this.useRecords===!1||S<64&&!this.randomAccessStructure?D[k++]=S:S<256?(D[k++]=204,D[k++]=S):S<65536?(D[k++]=205,D[k++]=S>>8,D[k++]=S&255):(D[k++]=206,xe.setUint32(k,S),k+=4);else if(S>>0===S)S>=-32?D[k++]=256+S:S>=-128?(D[k++]=208,D[k++]=S+256):S>=-32768?(D[k++]=209,xe.setInt16(k,S),k+=2):(D[k++]=210,xe.setInt32(k,S),k+=4);else{let N;if((N=this.useFloat32)>0&&S<4294967296&&S>=-2147483648){D[k++]=202,xe.setFloat32(k,S);let U;if(N<4||(U=S*hc[(D[k]&127)<<1|D[k+1]>>7])>>0===U){k+=4;return}else k--}D[k++]=203,xe.setFloat64(k,S),k+=8}else if(O==="object"||O==="function")if(!S)D[k++]=192;else{if(o){let U=o.get(S);if(U){if(!U.id){let L=o.idsToInsert||(o.idsToInsert=[]);U.id=L.push(U)}D[k++]=214,D[k++]=112,xe.setUint32(k,U.id),k+=4;return}else o.set(S,{offset:k-n})}let N=S.constructor;if(N===Object)x(S);else if(N===Array)E(S);else if(N===Map)if(this.mapAsEmptyObject)D[k++]=128;else{R=S.size,R<16?D[k++]=128|R:R<65536?(D[k++]=222,D[k++]=R>>8,D[k++]=R&255):(D[k++]=223,xe.setUint32(k,R),k+=4);for(let[U,L]of S)b(U),b(L)}else{for(let U=0,L=xh.length;U<L;U++){let j=Sh[U];if(S instanceof j){let V=xh[U];if(V.write){V.type&&(D[k++]=212,D[k++]=V.type,D[k++]=0);let Ce=V.write.call(this,S);Ce===S?Array.isArray(S)?E(S):x(S):b(Ce);return}let Y=D,J=xe,de=k;D=null;let ie;try{ie=V.pack.call(this,S,Ce=>(D=Y,Y=null,k+=Ce,k>Qe&&T(k),{target:D,targetView:xe,position:k-Ce}),b)}finally{Y&&(D=Y,xe=J,k=de,Qe=D.length-10)}ie&&(ie.length+k>Qe&&T(ie.length+k),k=F1(ie,D,k,V.type));return}}if(Array.isArray(S))E(S);else{if(S.toJSON){let U=S.toJSON();if(U!==S)return b(U)}if(O==="function")return b(this.writeFunction&&this.writeFunction(S));x(S)}}}else if(O==="boolean")D[k++]=S?195:194;else if(O==="bigint"){if(S<9223372036854776e3&&S>=-9223372036854776e3)D[k++]=211,xe.setBigInt64(k,S);else if(S<18446744073709552e3&&S>0)D[k++]=207,xe.setBigUint64(k,S);else if(this.largeBigIntToFloat)D[k++]=203,xe.setFloat64(k,Number(S));else{if(this.largeBigIntToString)return b(S.toString());if(this.useBigIntExtension||this.moreTypes){let N=S<0?BigInt(-1):BigInt(0),U;if(S>>BigInt(65536)===N){let L=BigInt(18446744073709552e3)-BigInt(1),j=[];for(;j.push(S&L),S>>BigInt(63)!==N;)S>>=BigInt(64);U=new Uint8Array(new BigUint64Array(j).buffer),U.reverse()}else{let L=S<0,j=(L?~S:S).toString(16);if(j.length%2?j="0"+j:parseInt(j.charAt(0),16)>=8&&(j="00"+j),bi)U=Buffer.from(j,"hex");else{U=new Uint8Array(j.length/2);for(let V=0;V<U.length;V++)U[V]=parseInt(j.slice(V*2,V*2+2),16)}if(L)for(let V=0;V<U.length;V++)U[V]=~U[V]}U.length+k>Qe&&T(U.length+k),k=F1(U,D,k,66);return}else throw new RangeError(S+" was too large to fit in MessagePack 64-bit integer format, use useBigIntExtension, or set largeBigIntToFloat to convert to float-64, or set largeBigIntToString to convert to string")}k+=8}else if(O==="undefined")this.encodeUndefinedAsNil?D[k++]=192:(D[k++]=212,D[k++]=0,D[k++]=0);else throw new Error("Unknown type: "+O)},C=this.variableMapSize||this.coercibleKeyAsNumber||this.skipValues?S=>{let O;if(this.skipValues){O=[];for(let U in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(U))&&!this.skipValues.includes(S[U])&&O.push(U)}else O=Object.keys(S);let R=O.length;R<16?D[k++]=128|R:R<65536?(D[k++]=222,D[k++]=R>>8,D[k++]=R&255):(D[k++]=223,xe.setUint32(k,R),k+=4);let N;if(this.coercibleKeyAsNumber)for(let U=0;U<R;U++){N=O[U];let L=Number(N);b(isNaN(L)?N:L),b(S[N])}else for(let U=0;U<R;U++)b(N=O[U]),b(S[N])}:S=>{D[k++]=222;let O=k-n;k+=2;let R=0;for(let N in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(N))&&(b(N),b(S[N]),R++);if(R>65535)throw new Error('Object is too large to serialize with fast 16-bit map size, use the "variableMapSize" option to serialize this object');D[O+++n]=R>>8,D[O+n]=R&255},v=this.useRecords===!1?C:t.progressiveRecords&&!p?S=>{let O,R=s.transitions||(s.transitions=Object.create(null)),N=k++-n,U;for(let L in S)if(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(L)){if(O=R[L],O)R=O;else{let j=Object.keys(S),V=R;R=s.transitions;let Y=0;for(let J=0,de=j.length;J<de;J++){let ie=j[J];O=R[ie],O||(O=R[ie]=Object.create(null),Y++),R=O}N+n+1==k?(k--,M(R,j,Y)):P(R,j,N,Y),U=!0,R=V[L]}b(S[L])}if(!U){let L=R[Ei];L?D[N+n]=L:P(R,Object.keys(S),N,0)}}:S=>{let O,R=s.transitions||(s.transitions=Object.create(null)),N=0;for(let L in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(L))&&(O=R[L],O||(O=R[L]=Object.create(null),N++),R=O);let U=R[Ei];U?U>=96&&p?(D[k++]=((U-=96)&31)+96,D[k++]=U>>5):D[k++]=U:M(R,R.__keys__||Object.keys(S),N);for(let L in S)(typeof S.hasOwnProperty!="function"||S.hasOwnProperty(L))&&b(S[L])},w=typeof this.useRecords=="function"&&this.useRecords,x=w?S=>{w(S)?v(S):C(S)}:v,T=S=>{let O;if(S>16777216){if(S-n>L1)throw new Error("Packed buffer would be larger than maximum buffer size");O=Math.min(L1,Math.round(Math.max((S-n)*(S>67108864?1.25:2),4194304)/4096)*4096)}else O=(Math.max(S-n<<2,D.length-1)>>12)+1<<12;let R=new mc(O);return xe=R.dataView||(R.dataView=new DataView(R.buffer,0,O)),S=Math.min(S,D.length),D.copy?D.copy(R,0,n,S):R.set(D.slice(n,S)),k-=n,n=0,Qe=R.length-10,D=R},M=(S,O,R)=>{let N=s.nextId;N||(N=64),N<d&&this.shouldShareStructure&&!this.shouldShareStructure(O)?(N=s.nextOwnId,N<m||(N=d),s.nextOwnId=N+1):(N>=m&&(N=d),s.nextId=N+1);let U=O.highByte=N>=96&&p?N-96>>5:-1;S[Ei]=N,S.__keys__=O,s[N-64]=O,N<d?(O.isShared=!0,s.sharedLength=N-63,i=!0,U>=0?(D[k++]=(N&31)+96,D[k++]=U):D[k++]=N):(U>=0?(D[k++]=213,D[k++]=114,D[k++]=(N&31)+96,D[k++]=U):(D[k++]=212,D[k++]=114,D[k++]=N),R&&(y+=I*R),g.length>=h&&(g.shift()[Ei]=0),g.push(S),b(O))},P=(S,O,R,N)=>{let U=D,L=k,j=Qe,V=n;D=Ms,k=0,n=0,D||(Ms=D=new mc(8192)),Qe=D.length-10,M(S,O,N),Ms=D;let Y=k;if(D=U,k=L,Qe=j,n=V,Y>1){let J=k+Y-1;J>Qe&&T(J);let de=R+n;D.copyWithin(de+Y,de+1,k),D.set(Ms.slice(0,Y),de),k=J}else D[R+n]=Ms[0]},F=S=>{let O=Wk(S,D,n,k,s,T,(R,N,U)=>{if(U)return i=!0;k=N;let L=D;return b(R),_(),L!==D?{position:k,targetView:xe,target:D}:k},this);if(O===0)return x(S);k=O}}useBuffer(t){D=t,D.dataView||(D.dataView=new DataView(D.buffer,D.byteOffset,D.byteLength)),xe=D.dataView,k=0}set position(t){k=t}get position(){return k}clearSharedData(){this.structures&&(this.structures=[]),this.typedStructs&&(this.typedStructs=[])}};Sh=[Date,Set,Error,RegExp,ArrayBuffer,Object.getPrototypeOf(Uint8Array.prototype).constructor,DataView,Cs];xh=[{pack(e,t,r){let n=e.getTime()/1e3;if((this.useTimestamp32||e.getMilliseconds()===0)&&n>=0&&n<4294967296){let{target:i,targetView:s,position:o}=t(6);i[o++]=214,i[o++]=255,s.setUint32(o,n)}else if(n>0&&n<4294967296){let{target:i,targetView:s,position:o}=t(10);i[o++]=215,i[o++]=255,s.setUint32(o,e.getMilliseconds()*4e6+(n/1e3/4294967296>>0)),s.setUint32(o+4,n)}else if(isNaN(n)){if(this.onInvalidDate)return t(0),r(this.onInvalidDate());let{target:i,targetView:s,position:o}=t(3);i[o++]=212,i[o++]=255,i[o++]=255}else{let{target:i,targetView:s,position:o}=t(15);i[o++]=199,i[o++]=12,i[o++]=255,s.setUint32(o,e.getMilliseconds()*1e6),s.setBigInt64(o+4,BigInt(Math.floor(n)))}}},{pack(e,t,r){if(this.setAsEmptyObject)return t(0),r({});let n=Array.from(e),{target:i,position:s}=t(this.moreTypes?3:0);this.moreTypes&&(i[s++]=212,i[s++]=115,i[s++]=0),r(n)}},{pack(e,t,r){let{target:n,position:i}=t(this.moreTypes?3:0);this.moreTypes&&(n[i++]=212,n[i++]=101,n[i++]=0),r([e.name,e.message,e.cause])}},{pack(e,t,r){let{target:n,position:i}=t(this.moreTypes?3:0);this.moreTypes&&(n[i++]=212,n[i++]=120,n[i++]=0),r([e.source,e.flags])}},{pack(e,t){this.moreTypes?yh(e,16,t):vh(bi?Buffer.from(e):new Uint8Array(e),t)}},{pack(e,t){let r=e.constructor;r!==z1&&this.moreTypes?yh(e,gh.indexOf(r.name),t):vh(e,t)}},{pack(e,t){this.moreTypes?yh(e,17,t):vh(bi?Buffer.from(e):new Uint8Array(e),t)}},{pack(e,t){let{target:r,position:n}=t(1);r[n]=193}}];function yh(e,t,r,n){let i=e.byteLength;if(i+1<256){var{target:s,position:o}=r(4+i);s[o++]=199,s[o++]=i+1}else if(i+1<65536){var{target:s,position:o}=r(5+i);s[o++]=200,s[o++]=i+1>>8,s[o++]=i+1&255}else{var{target:s,position:o,targetView:a}=r(7+i);s[o++]=201,a.setUint32(o,i+1),o+=4}s[o++]=116,s[o++]=t,e.buffer||(e=new Uint8Array(e)),s.set(new Uint8Array(e.buffer,e.byteOffset,e.byteLength),o)}function vh(e,t){let r=e.byteLength;var n,i;if(r<256){var{target:n,position:i}=t(r+2);n[i++]=196,n[i++]=r}else if(r<65536){var{target:n,position:i}=t(r+3);n[i++]=197,n[i++]=r>>8,n[i++]=r&255}else{var{target:n,position:i,targetView:s}=t(r+5);n[i++]=198,s.setUint32(i,r),i+=4}n.set(e,i)}function F1(e,t,r,n){let i=e.length;switch(i){case 1:t[r++]=212;break;case 2:t[r++]=213;break;case 4:t[r++]=214;break;case 8:t[r++]=215;break;case 16:t[r++]=216;break;default:i<256?(t[r++]=199,t[r++]=i):i<65536?(t[r++]=200,t[r++]=i>>8,t[r++]=i&255):(t[r++]=201,t[r++]=i>>24,t[r++]=i>>16&255,t[r++]=i>>8&255,t[r++]=i&255)}return t[r++]=n,t.set(e,r),r+=i,r}function Qk(e,t){let r,n=t.length*6,i=e.length-n;for(;r=t.pop();){let s=r.offset,o=r.id;e.copyWithin(s+n,s,i),n-=6;let a=s+n;e[a++]=214,e[a++]=105,e[a++]=o>>24,e[a++]=o>>16&255,e[a++]=o>>8&255,e[a++]=o&255,i=s}return e}function B1(e,t,r){if(_e.length>0){xe.setUint32(_e.position+e,k+r-_e.position-e),_e.stringsPosition=k-e;let n=_e;_e=null,t(n[0]),t(n[1])}}function eN(e,t){return e.isCompatible=r=>{let n=!r||(t.lastNamedStructuresLength||0)===r.length;return n||t._mergeStructures(r),n},e}var $1=new Ts({useRecords:!1}),tN=$1.pack,rN=$1.pack;var{NEVER:nN,ALWAYS:iN,DECIMAL_ROUND:sN,DECIMAL_FIT:oN}=dc,V1=512,G1=1024,j1=2048;var q1="0123456789ABCDEFGHJKMNPQRSTVWXYZ";function wh(e){let t=e.length,r=0,n="",i=0;for(let s=0;s<t;++s)for(r=r<<8|e[s],i+=8;i>=5;i-=5)n+=q1[r>>>i-5&31];return i>0&&(n+=q1[r<<5-i&31]),n}var yc=1,ks=Uint8Array.of(73,67,69,240,159,167,138,67,72,85,78,75),Y1=24,K1=ks.length+Y1+1+1+1;async function aN(e,t,r,n){if(e.byteLength<K1)throw new Error(`Expected icechunk header of ${K1} bytes, but received: ${e.byteLength} bytes`);let i=new DataView(e),s=0;for(let l=0,f=ks.length;l<f;++l)if(i.getUint8(l)!==ks[l])throw new Error(`Expected magic bytes of ${ks.join()} but received: ${new Uint8Array(e,0,f).join()}`);s+=ks.length,s+=Y1;let o=i.getUint8(s++);if(o>t)throw new Error(`Expected version <= ${t} but received: ${o}`);let a=i.getUint8(s++);if(a!==r)throw new Error(`Expected file type of ${r}, but received: ${a}`);let c=i.getUint8(s++),u=new Uint8Array(e,s);switch(c){case 0:break;case 1:u=await le(qr,n,[e],u),u=new Uint8Array(u.buffer,u.byteOffset,u.byteLength);break;default:throw new Error(`Unknown compression method: ${c}`)}return{content:u,specVersion:o}}async function vc(e,t,r,n){let{content:i,specVersion:s}=await aN(e,t,r,n);return{content:new Ar({mapsAsObjects:!1,int64AsType:"bigint"}).unpack(i),specVersion:s,estimatedSize:e.byteLength*3}}var J1=Je(oh([Is(Uint8Array)]),Ye(e=>e[0])),Ns=Je(J1,bs(12),Ye(wh)),cN=Je(J1,bs(8),Ye(wh)),uN=BigInt(Number.MIN_SAFE_INTEGER),lN=BigInt(Number.MAX_SAFE_INTEGER),fN=Je(rh(),eh(e=>e>=uN&&e<=lN,`Number outside supported range: [${Number.MIN_SAFE_INTEGER}, ${Number.MAX_SAFE_INTEGER}]`),Ye(Number)),$t=Hr([fN,Je(nh(),th())]);function mt(e){let t=Object.keys(e);return Je(ot(_t()),bs(t.length),Ye(r=>Object.fromEntries(t.map((n,i)=>[n,r[i]]))),lr(e))}var Ds=Ns,H1=Ns,xc=cN;function Sc(e,t,r){try{return{...y1(e,r.content),estimatedSize:r.estimatedSize}}catch(n){throw p1(n)?new Error(`Error parsing icechunk ${t}: ${JSON.stringify(m1(n.issues))}`):n}}var hN=2,pN=lr({Inline:Is(Uint8Array)}),dN=_t(),mN=Ue(),gN=mt({location:mN,offset:$t,length:$t,chunksum:dN}),yN=lr({Virtual:gN}),vN=mt({id:H1,offset:$t,length:$t}),xN=lr({Ref:vN}),SN=Je(Xt(Ue(),_t()),Ye(Object.fromEntries),Hr([pN,yN,xN])),wN=mt({id:Ds,chunks:Xt(xc,Xt(Je(ot($t),Ye(e=>e.join())),SN))});async function W1(e,t){let r=await vc(e,yc,hN,t);return Sc(wN,"chunk manifest",r)}function X1(e,t){return Ve(e,`manifests/${t}`)}function Z1(e){if(Ee(e),Object.keys(e).length!==1)throw new Error(`Expected object with only a "snapshot" property, but received: ${JSON.stringify(e)}`);let t=re(e,"snapshot",et);if(!Ps(t))throw new Error(`Expected icechunk snapshot id but received: ${JSON.stringify(t)}`);return t}function Ps(e){return e.match(/^[0-9ABCDEFGHJKMNPQRSTVWXYZ]{20}$/)!==null}function Q1(e){return e.match(/^[0-9ABCDEFGHJKMNPQRSTVWXYZ]{8}\.json$/)!==null}var EN=1,eS=Ns,bN=Ns,IN=mt({id:Ds,sizeBytes:$t,numRows:$t}),_N=mt({id:bN}),CN=Je(Xt(Ue(),_t()),Ye(Object.fromEntries),Hr([lr({Inline:_t()})])),AN=uc(["Slash","Dot"]),rS=Xt(Ue(),_t()),MN=mt({name:Ue(),configuration:rS}),TN=Je(Xt(Ue(),_t()),Ye(e=>{let t=Array.from(e.values());if(t.length!==1)throw new Error(`Expected a single key, but received: ${JSON.stringify(Array.from(e.keys()))}`);return t[0]})),kN=mt({name:Ue(),configuration:rS}),NN=ot(_s(Ue())),DN=mt({shape:ot($t),dataType:Ue(),chunkShape:ot($t),chunkKeyEncoding:AN,fillValue:TN,codecs:ot(MN),storageTransformers:ot(kN),dimensionNames:_s(NN)}),tS=ot($t),PN=sh([tS,tS]),RN=mt({objectId:Ds,extents:PN}),ON=uc(["Group"]),UN=lr({Array:mt({metadata:DN,manifests:ot(RN)})}),LN=Hr([ON,Je(Xt(Ue(),_t()),Ye(Object.fromEntries),UN)]),FN=mt({id:xc,path:Je(Ue(),Ye(e=>e==="/"?"":e.slice(1)+"/")),userAttributes:CN,nodeData:LN}),BN=Je(Xt(Ue(),FN),Ye(e=>Array.from(e.values()).sort((t,r)=>St(t.path,r.path)))),zN=mt({id:eS,parentId:_s(eS),flushedAt:Ue(),message:Ue(),metadata:ih(Ue(),_t()),manifestFiles:Je(ot(IN),Ye(e=>{let t=new Map;for(let r of e)t.set(r.id,r);return t})),attributeFiles:ot(_N),nodes:BN});async function nS(e,t){let r=await vc(e,yc,EN,t);return Sc(zN,"snapshot",r)}function iS(e){let{userAttributes:t,nodeData:r}=e,n;t===null?n=new Map:n=t.Inline;let i=r!=="Group"?$N(r.Array.metadata,n):{zarr_format:3,node_type:"group",attributes:n};return JSON.stringify(i,(s,o)=>o instanceof Map?Object.fromEntries(o):o)}function $N(e,t){let{shape:r,chunkShape:n,chunkKeyEncoding:i,dataType:s,fillValue:o,codecs:a,storageTransformers:c,dimensionNames:u}=e;return{zarr_format:3,node_type:"array",shape:r,data_type:s,chunk_grid:{name:"regular",configuration:{chunk_shape:n}},chunk_key_encoding:{name:"default",configuration:{separator:i==="Dot"?".":"/"}},fill_value:o,codecs:a,storage_transformers:c,dimension_names:u??void 0,attributes:t}}function sS(e,t){let{nodes:r}=e,n=er(r,t,(i,s)=>St(i,s.path));if(n<0)throw new Error(`Node not found: ${JSON.stringify(t)}`);return r[n]}function oS(e,t){return Ve(e,`snapshots/${t}`)}var aS=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),cS=e=>{throw TypeError(e)},uS=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&cS("Object expected");var n,i;r&&(n=t[aS("asyncDispose")]),n===void 0&&(n=t[aS("dispose")],r&&(i=n)),typeof n!="function"&&cS("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},lS=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,u){return u=Error(c),u.name="SuppressedError",u.error=o,u.suppressed=a,u},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};function Eh(e,t,r){let n=new qe(e.chunkManager.addRef(),{get:async(i,s)=>{let o=await e.kvStoreContext.read(i,{...s,throwIfMissing:!0});try{return await r(o.response,s.signal)}catch(a){throw new Error(`Error reading icechunk ${t} from ${i}`,{cause:a})}}});return n.registerDisposer(e.addRef()),n}function fS(e,t,r,n){return e.chunkManager.memoize.get("icechunk:snapshot",()=>Eh(e,"snapshot",async(s,o)=>{let a=await nS(await s.arrayBuffer(),o);return{data:a,size:a.estimatedSize}})).get(oS(t,r),n)}function hS(e,t,r,n){return e.chunkManager.memoize.get("icechunk:manifest",()=>Eh(e,"manifest",async(s,o)=>{let a=await W1(await s.arrayBuffer(),o);return{data:a,size:a.estimatedSize}})).get(X1(t,r),n)}function pS(e,t,r){return e.chunkManager.memoize.get("icechunk:ref",()=>Eh(e,"ref",async i=>({data:Z1(await i.json()),size:0}))).get(t,r)}function VN(e,t,r){return e.chunkManager.memoize.get("icechunk:branch",()=>{let i=new qe(e.chunkManager.addRef(),{get:async(s,o)=>{var a=[];try{let l=uS(a,new He(o.progressListener,{message:`Resolving icechunk branch at ${s}`}));try{let h=(await e.kvStoreContext.list(s,{...o,responseKeys:"suffix"})).entries.find(d=>Q1(d.key));if(h===void 0)throw new Error("Failed to find any refs");return{data:await pS(e,Ve(s,h.key),o),size:0}}catch(f){throw new Error(`Error resolving icechunk branch at ${s}`,{cause:f})}}catch(l){var c=l,u=!0}finally{lS(a,c,u)}}});return i.registerDisposer(e.addRef()),i}).get(t,r)}function GN(e,t,r){return e.chunkManager.memoize.get("icechunk:tag",()=>{let i=new qe(e.chunkManager.addRef(),{get:async(s,o)=>{var a=[];try{let l=uS(a,new He(o.progressListener,{message:`Resolving icechunk tag at ${s}`}));try{let[f,h]=await Promise.all([pS(e,Ve(s,"ref.json"),o),e.kvStoreContext.stat(Ve(s,"ref.json.deleted"),o)]);if(h!==void 0)throw new Error("Tag is marked as deleted");return{data:f,size:0}}catch(f){throw new Error(`Error resolving icechunk tag at ${s}`,{cause:f})}}catch(l){var c=l,u=!0}finally{lS(a,c,u)}}});return i.registerDisposer(e.addRef()),i}).get(t,r)}function dS(e,t,r,n){return"snapshot"in r?Promise.resolve(r.snapshot):"branch"in r?VN(e,Ve(t,`refs/branch.${r.branch}/`),n):GN(e,Ve(t,`refs/tag.${r.tag}/`),n)}function mS(e,t){let r,n,i=t.match(/(?:^|\/)(zarr\.json)$/);if(i!==null)r=t.slice(0,-i[1].length);else{let u=t.match(/c(?:[./][0-9]+)*$/);if(u===null)return;r=t.slice(0,-u[0].length);let l=u[0].split(/[./]/),f=l.length-1;n=new Array(f);for(let h=0;h<f;++h)n[h]=Number(l[h+1])}let s=sS(e,r);if(n===void 0)return{node:s};if(s.nodeData==="Group")return;let{shape:o,chunkShape:a}=s.nodeData.Array.metadata,c=o.length;if(c===n.length){for(let u=0;u<c;++u)if(n[u]*a[u]>=o[u])return;return{node:s,chunk:n}}}function jN([e,t],r){for(let n=0,i=r.length;n<i;++n){let s=r[n];if(s<e[n]||s>=t[n])return!1}return!0}async function gS(e,t,r,n,i){let{manifests:s}=r.nodeData.Array,o=n.join(),a=r.id;for(let c of s){if(!jN(c.extents,n))continue;let l=(await hS(e,t,c.objectId,i)).chunks.get(a);if(l===void 0)continue;let f=l.get(o);if(f!==void 0)return f}}async function yS(e,t,r,n,i){let s=mS(r,n);if(s===void 0)return;let{node:o,chunk:a}=s;if(a===void 0)return{totalSize:void 0};let c=await gS(e,t,o,a,i);if(c===void 0)return;let u;return"Inline"in c?u=c.Inline.length:"Virtual"in c?u=c.Virtual.length:u=c.Ref.length,{totalSize:u}}async function qN(e,t,r,n){if("Inline"in r)return ls(r.Inline,n.byteRange);let i,s,o;if("Virtual"in r)({location:o,offset:i,length:s}=r.Virtual);else{let{Ref:a}=r;({offset:i,length:s}=a),o=KN(t,a.id)}return new tt(e.kvStoreContext.getFileHandle(o),{offset:i,length:s}).read(n)}function KN(e,t){return Ve(e,`chunks/${t}`)}async function vS(e,t,r,n,i){let s=mS(r,n);if(s===void 0)return;let{node:o,chunk:a}=s;if(a===void 0){let u=iS(o),l=new TextEncoder().encode(u);return ls(l,i.byteRange)}let c=await gS(e,t,o,a,i);if(c!==void 0)return qN(e,t,c,i)}var bh="branch.",Ih="tag.";function SS(e,t){let{baseUrl:r,refSpec:n}=e,i=n===void 0?"":`@${YN(n)}/`;return r+`|icechunk:${i}${Ze(t)}`}function YN(e){return"branch"in e?bh+Ze(e.branch):"tag"in e?Ih+Ze(e.tag):e.snapshot}function xS(e){return e.length>0&&!e.includes("/")}function _h(e){if(e!==void 0){if(e.startsWith(bh)){let t=e.substring(bh.length);if(!xS(t))throw new Error(`Invalid branch name: ${JSON.stringify(t)}`);return{branch:decodeURIComponent(t)}}if(e.startsWith(Ih)){let t=e.substring(Ih.length);if(!xS(t))throw new Error(`Invalid tag name: ${JSON.stringify(t)}`);return{tag:decodeURIComponent(t)}}if(Ps(e))return{snapshot:e};throw new Error(`Invalid ref spec: ${JSON.stringify(e)}`)}}function wS(e,t){Rr(e);try{let r=(e.suffix??"").match(/^(?:@([^/]*)(?:\/|$))?(.*)$/),[,n,i]=r;return{baseUrl:t.store.getUrl(nn(t.path)),version:_h(n),path:decodeURIComponent(i)}}catch(r){throw new Error(`Invalid URL: ${e.url}`,{cause:r})}}var wc=class{constructor(t,r,n){this.sharedKvStoreContext=t,this.baseUrl=r,this.refSpec=n}snapshot;async getSnapshot(t){let{snapshot:r}=this;if(r===void 0){let n=await dS(this.sharedKvStoreContext,this.baseUrl,this.refSpec??{branch:"main"},t);r=this.snapshot=await fS(this.sharedKvStoreContext,this.baseUrl,n,t)}return r}getUrl(t){return SS(this,t)}async stat(t,r){let n=await this.getSnapshot(r);return yS(this.sharedKvStoreContext,this.baseUrl,n,t,r)}async read(t,r){let n=await this.getSnapshot(r);return vS(this.sharedKvStoreContext,this.baseUrl,n,t,r)}async list(t,r){let n=await this.getSnapshot(r);return a1(n,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};async function ES(e,t){let{url:r}=t,n=r.suffix??"";if(n==="")return{offset:0,completions:[{value:"@",description:"Ref specifier"}]};let i=n.match(/^@([^/]*)((?:\/|$).*)/);if(i===null)return;let[,s,o]=i;if(o!==""){_h(s);return}let a;if(s.match(/^(?:(?:(?:t|$)(?:a|$)(?:g|$)(?:\.|$))|(?:(?:b|$)(?:r|$)(?:a|$)(?:n|$)(?:c|$)(?:h|$)(?:\.|$)))/)){let u=el(t.base.path,"refs/");a=Xi(t.base.store,u+decodeURIComponent(s),{signal:t.signal,progressListener:t.progressListener}).then(({directories:l})=>l.map(f=>{let h=f.slice(u.length);return{value:Ze(h)+"/",description:h.startsWith("tag.")?"Tag":"Branch"}}))}let c;if(s.match(/^[0-9ABCDEFGHJKMNPQRSTVWXYZ]{0,20}$/)){let u=el(t.base.path,"snapshots/");c=Xi(t.base.store,u+s,{signal:t.signal,progressListener:t.progressListener}).then(({entries:l})=>{let f=[];for(let{key:h}of l){let p=h.slice(u.length);Ps(p)&&f.push({value:p+"/",description:"Snapshot"})}return f})}return{offset:1,completions:[...await a??[],...await c??[]]}}function JN(e){return{scheme:"icechunk",description:"Icechunk repository",getKvStore(t,r){let{baseUrl:n,version:i,path:s}=wS(t,r);return{store:new wc(e,n,i),path:s}},completeUrl(t){return ES(e,t)}}}wt.registerKvStoreAdapterProvider(JN);var Ch="middleauth+";function HN(e,t){return e.getCredentialsProvider("middleauthapp",new URL(t).origin)}function WN(e,t,r){return{scheme:Ch+e,description:`${e} with middleauth`,getKvStore(n){let i=n.url.substring(Ch.length),s=HN(t.credentialsManager,i);try{let{baseUrl:o,path:a}=Dn(i);return{store:new r(t,o,Ch+o,ea(s)),path:a}}catch(o){throw new Error(`Invalid URL ${JSON.stringify(n.url)}`,{cause:o})}}}}function bS(e,t){for(let r of["https"])e.registerBaseKvStoreProvider(n=>WN(r,n,t))}bS(wt,wi);function XN(e,t,r){return Ss?e.getCredentialsProvider("gcs",{bucket:r}):e.getCredentialsProvider("ngauth_gcs",{authServer:t,bucket:r})}var IS="gs+ngauth+";function ZN(e,t){return{scheme:e,description:Ss?"Google Cloud Storage":"Google Cloud Storage (ngauth)",getKvStore(r){let n=(r.suffix??"").match(/^\/\/([^/]+)\/([^/]+)(\/.*)?$/);if(n===null)throw new Error(`Invalid URL, expected ${r.scheme}://<ngauth-server>/<bucket>/<path>`);let[,i,s,o]=n,a=r.scheme.substring(IS.length)+"://"+i,c=XN(t.credentialsManager,a,s);return{store:new Si(s,`${r.scheme}://${i}/${s}/`,ea(c)),path:decodeURIComponent((o??"").substring(1))}}}}for(let e of["http","https"])Wt.registerBaseKvStoreProvider(t=>ZN(`${IS}${e}`,t));var TS=Ri(CS(),1);function AS(e,t){let r=0,n=0;for(let i=t,s=e.byteLength;i<s;++i){let o=e.getUint8(i);if(r+=(o&127)<<n,(o&128)===0){if(r>Number.MAX_SAFE_INTEGER)throw new Error(`Value exceeded ${Number.MAX_SAFE_INTEGER}`);return{offset:i+1,value:r}}n+=7}throw new Error("Unexpected EOF")}function MS(e,t){let r=0n,n=0n;for(let i=t,s=e.byteLength;i<s;++i){let o=e.getUint8(i);if(r|=BigInt(o&127)<<BigInt(n),(o&128)===0)return{offset:i+1,value:r};n+=7n}throw new Error("Unexpected EOF")}var Ec=(e=>(e[e.UNCOMPRESSED=0]="UNCOMPRESSED",e[e.ZSTD=1]="ZSTD",e))(Ec||{});async function Ii(e,t,r,n){if(e.byteLength<18)throw new Error("Unexpected EOF");let i=new DataView(e),s=i.getUint32(0,!1);if(s!==t)throw new Error(`Expected magic value 0x${t.toString(16)} but received 0x${s.toString(16)}`);let o=i.getBigUint64(4,!0);if(o!=BigInt(e.byteLength))throw new Error(`Expected length ${e.byteLength} but received: ${o}`);let a=i.getUint32(e.byteLength-4,!0),c=(0,TS.buf)(new Uint8Array(e,0,e.byteLength-4))>>>0;if(a!=c)throw new Error(`Expected CRC32c checksum of ${a}, but received ${c}`);let u=i.getUint8(12);if(u>r)throw new Error(`Expected version to be <= ${r}, but received: ${u}`);let l=i.getUint8(13),f=new Uint8Array(e,14,e.byteLength-14-4);switch(l){case 0:break;case 1:f=await le(qr,n,[e],f);break;default:throw new Error(`Unknown compression format ${l}`)}return{reader:{offset:0,data:new DataView(f.buffer,f.byteOffset,f.byteLength)},version:u}}function hr(e,t){let{offset:r,data:n}=e;if(r+t>n.byteLength)throw new Error("Unexpected EOF");return e.offset+=t,new Uint8Array(n.buffer,n.byteOffset+r,t)}function Zt(e){let{value:t,offset:r}=AS(e.data,e.offset);return e.offset=r,t}function Ct(e){let{value:t,offset:r}=MS(e.data,e.offset);return e.offset=r,t}function xn(e,t){let r=Zt(e);if(r>t)throw new Error(`Expected value <= ${t}, but received: ${r}`);return r}function Sn(e){let{offset:t,data:r}=e;if(t+1>r.byteLength)throw new Error("Unexpected EOF");return e.offset+=1,r.getUint8(t)}function kS(e){let{offset:t,data:r}=e;if(t+4>r.byteLength)throw new Error("Unexpected EOF");return e.offset+=4,r.getInt32(t,!0)}function Mh(e){let{offset:t,data:r}=e;if(t+8>r.byteLength)throw new Error("Unexpected EOF");return e.offset+=8,r.getBigUint64(t,!0)}function NS(e){if(e.offset!==e.data.byteLength)throw new Error(`Expected EOF at byte ${e.offset}`)}function Be(e){return(t,r,n)=>{let i=[];for(let s=0;s<r;++s)i[s]=e(t,n);return i}}function bc(e,t){let r=Object.keys(t),n=[];for(let i=0;i<e;++i){let s=Object.fromEntries(r.map(o=>[o,t[o][i]]));n[i]=s}return n}function Zr(e,t){return(r,n,i)=>{let s=Object.fromEntries(Object.entries(e).map(([a,c])=>[a,c(r,n,i)])),o=bc(n,s);if(t!==void 0)for(let a=0;a<n;++a)t?.(o[a],i);return o}}var pr=new Uint8Array(0);function Mr(e,t){let r=Math.min(e.length,t.length);for(let n=0;n<r;++n){let i=e[n]-t[n];if(i!==0)return i}return e.length-t.length}function Rs(e,t){let r=Math.min(e.length,t.length);for(let n=0;n<r;++n){let i=e[n]-t[n];if(i!==0)return{offset:n,difference:i}}return{offset:r,difference:e.length-t.length}}var S7={inclusiveMin:pr,exclusiveMax:Uint8Array.of(0)};function _i(...e){let t=0;for(let i of e)t+=i.length;let r=new Uint8Array(t),n=0;for(let i of e)r.set(i,n),n+=i.length;return r}function Ic(e,t){return e.length>=t.length&&Rs(e,t).offset===t.length}function kh(e,t){let{dataFileTable:r}=t,n=Zt(e);if(n>=r.length)throw new Error(`Invalid data file index ${n}, expected value <= ${r.length}`);return r[n]}var _c=Zr({dataFile:Be(kh),offset:Be(Ct),length:Be(Ct)},(e,t)=>{if(wn(e)){if(t.allowMissing!==!0)throw new Error("Reference to missing value not allowed")}else if(e.offset+e.length>BigInt(Number.MAX_SAFE_INTEGER))throw new Error(`Offset=${e.offset} + length=${e.length} exceeds maximum of ${Number.MAX_SAFE_INTEGER}`)});function wn(e){return e.offset===0xffffffffffffffffn&&e.length===0xffffffffffffffffn}var Th=65535;function Ci(e,t){let r=Zt(e),n=new Uint16Array(r*3);for(let c=1,u=r*3;c<u;++c)n[c]=xn(e,Th);let i=[],s=pr,o=pr,a=new TextDecoder("utf-8",{fatal:!0});for(let c=0;c<r;++c){let u=n[c],l=n[c+r],f=n[c+2*r],h=u+l;if(h>Th)throw new Error(`path_length[${c} = prefix_length(${u}) + suffix_length(${l}) = ${h} > ${Th}`);if(f>h)throw new Error(`base_path_length[${c}] = ${f} > path_length(${h}) = prefix_length(${u}) + suffix_length(${l})`);if(u>Math.min(s.length,f)&&f!==s.length)throw new Error(`path_prefix_length[${c-1}] = ${u} > min(base_path_length[${c-1}] = ${s.length}, base_path_length[${c}] = ${f}) is not valid if base_path_length[${c-1}] != base_path_length[${c}]`);let p=u+l-f,d,m;if(f===0)d=t,s=pr;else if(u>=f)d=i[c-1].baseUrl;else{let g=new Uint8Array(f),y=0,I=Math.max(f-u,0);if(u>0){let _=Math.min(u,f);g.set(s.subarray(0,_)),y=_,u-=_}I!==0&&(g.set(hr(e,I),y),l-=I),d=Ve(t,a.decode(g)),s=g}if(p===0)m="",o=pr;else if(l===0&&p===o.length)m=i[c-1].relativePath;else{let g=new Uint8Array(p),y=0;u!==0&&(g.set(o.subarray(0,u),0),y+=u),l>0&&g.set(hr(e,l),y),m=a.decode(g),o=g}i[c]={baseUrl:d,relativePath:m}}return i}var QN=215687390,eD=0,DS=1024*1024;async function OS(e,t,r){try{let{reader:n}=await Ii(e,QN,eD,r),i=Sn(n),s=Ci(n,t),o=Zt(n);if(o===0)throw new Error("Empty b+tree node");if(o>DS)throw new Error(`B+tree node has arity ${o}, which exceeds limit of ${DS}`);return{height:i,...i===0?rD(n,s,o):nD(n,s,o),estimatedSize:n.data.byteLength*3}}catch(n){throw new Error("Error decoding OCDBT b+tree node",{cause:n})}}var Nh=65535;function PS(e){return xn(e,Nh)}function US(e,t,r){let n=new Uint16Array(t*2);for(let c=1,u=n.length;c<u;++c)n[c]=PS(e);let i=n[t];for(let c=1;c<t;++c)i=Math.min(i,n[c]);let s;if(r){s=new Uint16Array(t);for(let c=0;c<t;++c){let u=s[c]=PS(e);i=Math.min(i,u)}}i=Math.min(n[t],i);for(let c=0,u=0;c<t;++c){let l=n[c];if(l>u)throw new Error(`Child ${c}: Prefix length of ${l} exceeds previous key length ${u}`);let f=n[c+t],h=l+f;if(h>Nh)throw new Error(`Child ${c}: Key length ${h} exceeds limit of ${Nh}`);if(r){let p=s[c];if(p>h)throw new Error(`Child ${c}: subtree common prefix length of ${p} exceeds key length of ${h}`);s[c]-=i}u=h}let o=new Array(t),a;{let c=n[t],u=hr(e,c);a=u.slice(0,i),o[0]=u.slice(i)}for(let c=1;c<t;++c){let u=n[c]-i,l=n[c+t],f=hr(e,l),h=o[c-1];if(Mr(h.subarray(u),f)>=0)throw new Error("Invalid key order");let p=new Uint8Array(u+l);p.set(h.subarray(0,u)),p.set(f,u),o[c]=p}return{keys:o,subtreeCommonPrefixLengths:s,commonPrefix:a}}var RS=1024*1024;function tD(e,t,r){let n=Be(Ct)(e,r,{}),i=hr(e,r);for(let o=0;o<r;++o){let a=i[o];if(a>1)throw new Error(`value_kind[${o}]=${a} is outside valid range [0, 1]`);if(a===0){let c=n[o];if(c>BigInt(RS))throw new Error(`value_length[${o}]=${c} exceeds maximum of ${RS} for an inline value`)}}let s=new Array(r);for(let o=0;o<r;++o){if(i[o]!==1)continue;let a=kh(e,{dataFileTable:t});s[o]={dataFile:a,offset:0n,length:n[o]}}for(let o=0;o<r;++o){if(i[o]!==1)continue;let a=Ct(e);s[o].offset=a}for(let o=0;o<r;++o)i[o]===0&&(s[o]=hr(e,Number(n[o])));return s}function rD(e,t,r){let{keys:n,commonPrefix:i}=US(e,r,!1),s=tD(e,t,r);return{keyPrefix:i,entries:bc(r,{key:n,value:s})}}function nD(e,t,r){let{keys:n,commonPrefix:i,subtreeCommonPrefixLengths:s}=US(e,r,!0),o=Dh(e,r,{dataFileTable:t});return{keyPrefix:i,entries:bc(r,{key:n,subtreeCommonPrefixLength:s,node:o})}}var iD=Zr({numKeys:Be(Ct),numTreeBytes:Be(Ct),numIndirectValueBytes:Be(Ct)}),Dh=Zr({location:_c,statistics:iD});function Cc(e,t,r){if(e.height!==t)throw new Error(`Expected height of ${t} but received ${e.height}`);let{keyPrefix:n}=e;if(r.length<n.length){if(Mr(n,r)>=0)return}else if(Mr(n,r.subarray(0,n.length))>=0&&Mr(e.entries[0].key,r.subarray(n.length))>=0)return;throw new Error(`First key [${n}]+[${e.entries[0].key}] < inclusive_min [${r}] specified by parent node`)}function sD(e,t){let r=we(0,e.length,n=>Mr(e[n].key,t)>0);return Math.max(0,r-1)}function oD(e,t){return we(0,e.length,r=>Mr(e[r].key,t)>=0)}function LS(e,t){let r=sD(e,t),n=Os(e,r,e.length,t);return[r,n]}function Os(e,t,r,n){return t===r||n.length===0?r:we(t,r,i=>{let{offset:s,difference:o}=Rs(n,e[i].key);return o<0&&s<n.length})}function FS(e,t){let r=oD(e,t),n=Os(e,r,e.length,t);return[r,n]}function BS(e,t){let r=er(e,t,(n,i)=>Mr(n,i.key));if(!(r<0))return e[r]}function zS(e,t){let r=we(0,e.length,s=>Mr(e[s].key,t)>0);if(r===0)return;let n=e[r-1],{subtreeCommonPrefixLength:i}=n;if(!(i!==0&&!Ic(t,n.key.subarray(0,i))))return n}var $S=16;function Ph(e,t,r){let n=2**t,i=xn(e,n),s=dD(e,i,{allowMissing:!0,dataFileTable:r});return cD(s,t),s}function aD(e,t,r,n){let i=Ac(t);if(n>i)throw new Error(`height=${n} exceeds maximum of ${i} for version_tree_arity_log2=${t}`);let s=2**t,o=Uh(e,r,s,n-1);return uD(o,t,n),o}function cD(e,t){let r=2**t;if(e.length===0||e.length>r)throw new Error(`num_children=${e.length} outside valid range [1, ${r}]`);for(let[o,a]of e.entries()){if(wn(a.root.location)){if(a.rootHeight!==0)throw new Error(`non-zero root_height=${a.rootHeight} for empty generation ${a.generationNumber}`);let{statistics:c}=a.root;if(c.numKeys!==0n||c.numTreeBytes!==0n||c.numIndirectValueBytes!==0n)throw new Error(`non-zero statistics for empty generation_number[${o}]=${a.generationNumber}`)}if(a.generationNumber===0n)throw new Error(`generation_number[${o}] must be non-zero`);if(o!==0&&a.generationNumber<=e[o-1].generationNumber)throw new Error(`generation_number[${o}]=${a.generationNumber} <= generation_number[${o-1}]=${e[o-1].generationNumber}`)}let n=e.at(-1).generationNumber,i=e[0].generationNumber,s=Rh(t,0,n);if(i<s)throw new Error(`Generation range [${i}, ${n}] exceeds maximum of [${s}, ${n}]`)}function uD(e,t,r){let n=2**t;if(e.length===0||e.length>n)throw new Error(`num_children=${e.length} outside valid range [1, ${n}]`);let i=1n<<BigInt(t*r);for(let[a,c]of e.entries()){if(c.generationNumber===0n)throw new Error(`generation_number[${a}] must be non-zero`);if(a!==0){let u=e[a-1];if(c.generationNumber<=u.generationNumber)throw new Error(`generation_number[${a}]=${c.generationNumber} >= generation_number[${a-1}]=${u.generationNumber}`);if((c.generationNumber-1n)/i===(u.generationNumber-1n)/i)throw new Error(`generation_number[${a}]=${c.generationNumber} should be in the same child node as generation_number[${a-1}]=${u.generationNumber}`)}if(c.generationNumber%i!==0n)throw new Error(`generation_number[${a}]=${c.generationNumber} is not a multiple of ${i}`);if(c.numGenerations>i)throw new Error(`num_generations[${a}]=${c.numGenerations} for generation_number=${c.generationNumber} is greater than ${i}`)}let s=1n<<BigInt(t),o=e.at(-1);if((o.generationNumber-1n)/i/s!==(e[0].generationNumber-1n)/i/s)throw new Error(`generation_number[0]=${e[0].generationNumber} cannot be in the same node as generation_number[${e.length-1}]=${o.generationNumber}`)}function Rh(e,t,r){return r-(r-1n)%(1n<<BigInt(e*(t+1)))}function Oh(e){let t=Sn(e);if(t===0||t>$S)throw new Error(`Expected version_tree_arity_log2 in range [1, ${$S}] but received: ${t}`);return t}var lD=215683636,fD=0;async function VS(e,t,r){try{let{reader:n}=await Ii(e,lD,fD,r),i=Oh(n),s=Sn(n),o=Ci(n,t);return{versionTreeArityLog2:i,height:s,entries:s===0?Ph(n,i,o):aD(n,i,o,s),estimatedSize:n.data.byteLength*3}}catch(n){throw new Error("Error decoding OCDBT version tree node",{cause:n})}}var hD=Zr({generationNumber:Be(Ct),location:_c,numGenerations:Be(Ct),commitTime:Be(Mh),height:Be((e,{height:t})=>t===void 0?Sn(e):t),cumulativeNumGenerations:Be(()=>0n)});function pD(e){let t=0n;for(let r of e)t+=r.numGenerations,r.cumulativeNumGenerations=t}function Uh(e,t,r,n){let i=xn(e,r),s=hD(e,i,{dataFileTable:t,height:n});return pD(s),s}function Ac(e){return Math.floor(63/e)-1}var dD=Zr({generationNumber:Be(Ct),rootHeight:Be(Sn),root:Dh,commitTime:Be(Mh)});function Lh(e,t){return"generationNumber"in e?mr(e.generationNumber,t.generationNumber):mr(e.commitTime,t.commitTime)}function GS(e,t,r){if("generationNumber"in r)return mD(t,r.generationNumber);if("generationIndex"in r){let{generationIndex:n}=r;return n-=e,n<0n?-1:n>=BigInt(t.length)?t.length:Number(n)}else return gD(t,r.commitTime)}function mD(e,t){let r=er(e,t,(n,i)=>mr(n,i.generationNumber));return r<0?e.length:r}function gD(e,t){let r=we(0,e.length,n=>e[n].commitTime>t);return r===0?e.length:r-1}function Ai(e,t,r){if("generationIndex"in r){let n=r.generationIndex-e;return n<0n?0:n>BigInt(t.length)?t.length:Number(n)}return we(0,t.length,n=>Lh(r,t[n])<=0)}function jS(e,t,r,n){return"generationIndex"in n?r[qS(r,n.generationIndex-t)]:"generationNumber"in n?yD(e,r,n.generationNumber):vD(r,n.commitTime)}function qS(e,t){return we(0,e.length,r=>e[r].cumulativeNumGenerations>t)}function yD(e,t,r){let n=we(0,t.length,s=>t[s].generationNumber>=r);if(n===t.length)return;let i=t[n];if(!(Rh(e,i.height,i.generationNumber)>r))return i}function vD(e,t){let r=we(0,e.length,n=>e[n].commitTime>t);if(r!==0)return e[r-1]}function Mc(e,t,r,n){return"generationIndex"in n?qS(r,n.generationIndex-t):"generationNumber"in n?xD(e,r,n.generationNumber):KS(r,n.commitTime)}function xD(e,t,r){return we(0,t.length,n=>{let i=t[n];return Rh(e,i.height,i.generationNumber)>=r})}function KS(e,t){let r=we(0,e.length,n=>e[n].commitTime>t);return Math.max(0,r-1)}function Tc(e,t,r){return"generationIndex"in r?SD(t,r.generationIndex-e):"generationNumber"in r?wD(t,r.generationNumber):KS(t,r.commitTime)}function SD(e,t){return we(0,e.length,r=>{let n=e[r];return n.cumulativeNumGenerations-n.numGenerations>=t})}function wD(e,t){return we(0,e.length,r=>e[r].generationNumber>=t)}function kc(e,t,r,n,i){if(e.height!==n)throw new Error(`Expected height of ${n} but received: ${e.height}`);if(e.versionTreeArityLog2!==t.versionTreeArityLog2)throw new Error(`Expected version_tree_arity_log2=${t.versionTreeArityLog2} but received: ${e.versionTreeArityLog2}`);let{generationNumber:s}=e.entries.at(-1);if(s!==r)throw new Error(`Expected generation number ${r} but received: ${s}`);let o=e.height===0?BigInt(e.entries.length):e.entries.at(-1).cumulativeNumGenerations;if(o!==i)throw new Error(`Expected ${i}, but received: ${o}`)}function ED(e){let t=hr(e,16).slice(),r=Zt(e);if(r>1)throw new Error(`Unknown manifest kind: ${r}`);let n=Zt(e),i=Zt(e),s=Oh(e),o=Zt(e),a;switch(o){case Ec.UNCOMPRESSED:break;case Ec.ZSTD:a=kS(e);break;default:throw new Error(`Invalid compression method: ${o}`)}return{uuid:t,manifestKind:r,maxInlineValueBytes:n,maxDecodedNodeBytes:i,versionTreeArityLog2:s,compressionMethod:o,zstdLevel:a}}function bD(e,t,r){let n=Ci(e,r),i=Ph(e,t.versionTreeArityLog2,n),s=ID(e,t.versionTreeArityLog2,n,i.at(-1).generationNumber);return{inlineVersions:i,versionTreeNodes:s,numGenerations:BigInt(i.length)+(s.at(-1)?.cumulativeNumGenerations??0n)}}function ID(e,t,r,n){let i=Ac(t),s=Uh(e,r,i,void 0);return _D(t,n,s),s}function _D(e,t,r){let n=Ac(e);for(let[s,o]of r.entries()){if(o.height===0||o.height>n)throw new Error(`entry_height[${s}]=${o.height} outside valid range [1, ${n}]`);if(o.generationNumber===0n)throw new Error(`generation_number[${s}] must be non-zero`);if(s>0){let a=r[s-1];if(o.generationNumber<=a.generationNumber)throw new Error(`generation_number[${s}]=${o.generationNumber} <= generation_number[${s-1}]=${a.generationNumber}`);if(o.height>=a.height)throw new Error(`entry_height[${s}]=${o.height} >= entry_height[${s-1}]=${a.height}`)}}let i=r.length;for(let{minGenerationNumber:s,maxGenerationNumber:o,height:a}of CD(t,e)){if(i===0)break;let c=r[i-1];if(c.height!==a)continue;--i;let{generationNumber:u}=c;if(u<s||u>o)throw new Error(`generation_number[${i}]=${u} is outside expected range [${s}, ${o}] for height ${a}`)}if(i!==0)throw new Error(`Unexpected child with generation_number[${i-1}]=${r[i-1].generationNumber} and entry_height=${r[i-1].height} given last generation_number=${t}`)}function CD(e,t){e=e-1n>>BigInt(t)<<BigInt(t);let r=1,n=[];for(;e!==0n;){let i=BigInt((r+1)*t),s=e-1n>>i<<i,o=s+1n;n.push({minGenerationNumber:o,maxGenerationNumber:e,height:r}),++r,e=s}return n}var AD=215693866,MD=0;async function YS(e,t,r){try{let{reader:n}=await Ii(e,AD,MD,r),i=ED(n),s=i.manifestKind===0?bD(n,i,t):void 0;return NS(n),{config:i,versionTree:s,estimatedSize:n.data.byteLength*3}}catch(n){throw new Error("Error decoding OCDBT manifest",{cause:n})}}var JS=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),HS=e=>{throw TypeError(e)},TD=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&HS("Object expected");var n,i;r&&(n=t[JS("asyncDispose")]),n===void 0&&(n=t[JS("dispose")],r&&(i=n)),typeof n!="function"&&HS("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},kD=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,u){return u=Error(c),u.name="SuppressedError",u.error=o,u.suppressed=a,u},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};function ND(e,t,r){return e.chunkManager.memoize.get("ocdbt:manifest",()=>{let i=new qe(e.chunkManager.addRef(),{get:async(s,o)=>{var a=[];try{let l=Ve(s.baseUrl,s.relativePath),f=TD(a,new He(o.progressListener,{message:`Reading OCDBT manifest from ${l}`})),h=await e.kvStoreContext.read(l,{...o,throwIfMissing:!0});try{let p=await YS(await h.response.arrayBuffer(),s.baseUrl,o.signal);return{data:p,size:p.estimatedSize}}catch(p){throw new Error(`Error reading OCDBT manifest from ${l}`,{cause:p})}}catch(l){var c=l,u=!0}finally{kD(a,c,u)}}});return i.registerDisposer(e.addRef()),i}).get(t,r)}async function Us(e,t,r){let n=await ND(e,{baseUrl:t,relativePath:"manifest.ocdbt"},r);if(n.versionTree===void 0)throw new Error("only manifest_kind=single is supported");return n}function WS(e,t,r){let n=new qe(e.chunkManager.addRef(),{get:async(i,s)=>{let{dataFile:o}=i,a=Ve(o.baseUrl,o.relativePath),c=await e.kvStoreContext.read(a,{...s,throwIfMissing:!0,byteRange:{offset:Number(i.offset),length:Number(i.length)}});try{let u=await r(await c.response.arrayBuffer(),o.baseUrl,s.signal);return{data:u,size:u.estimatedSize}}catch(u){throw new Error(`Error reading OCDBT ${t} from ${a}`,{cause:u})}},encodeKey:({dataFile:i,offset:s,length:o})=>JSON.stringify([i,`${s}/${o}`])});return n.registerDisposer(e.addRef()),n}function Nc(e,t,r){return e.chunkManager.memoize.get("ocdbt:btree",()=>WS(e,"b+tree node",OS)).get(t,r)}function Dc(e,t,r){return e.chunkManager.memoize.get("ocdbt:versionnode",()=>WS(e,"version tree node",VS)).get(t,r)}var Fh=!1;async function XS(e,t,r,n){let i=[],s=new Set;wn(t.root.location)||await ZS(t.root,t.rootHeight,pr,0,{sharedKvStoreContext:e,prefix:r,entries:i,directories:s,signal:n.signal,progressListener:n.progressListener});let o=Eo({entries:i,directories:Array.from(s)});return Fh&&console.log(JSON.stringify(o)),o}async function ZS(e,t,r,n,i){i.signal?.throwIfAborted();let s=await Nc(i.sharedKvStoreContext,e.location,i);Cc(s,t,r.subarray(n));let o=_i(r.subarray(0,n),s.keyPrefix);Fh&&console.log("listSubtree",{nodeReference:e,height:t,inclusiveMinKey:r,subtreeCommonPrefixLength:n});let a=l=>{try{i.directories.add(new TextDecoder("utf-8",{fatal:!0}).decode(l))}catch{}},{prefix:c}=i;{let{offset:l,difference:f}=Rs(c,o);if(f!==0&&l<Math.min(c.length,o.length))return}if(c.length<o.length){let l=o.indexOf(47,c.length);if(l!==-1){a(o.subarray(0,l));return}}let u=c.subarray(o.length);if(s.height>0){let l=s.entries,[f,h]=LS(l,u);Fh&&console.log("Got entry range",f,h,l.length,u);let p=[];for(let d=f;d<h;){let m=l[d];++d;let{key:g}=m,{subtreeCommonPrefixLength:y}=m;if(y>u.length){let I=g.indexOf(47,u.length);if(I!==-1){let _=g.subarray(0,I);a(_i(o,_)),d=Os(l,d,h,_);continue}}p.push(ZS(m.node,t-1,_i(o,m.key),o.length+m.subtreeCommonPrefixLength,i))}await Promise.all(p)}else{let l=s.entries,[f,h]=FS(l,u);for(let p=f;p<h;){let d=l[p];++p;let{key:m}=d,g=m.indexOf(47,u.length);if(g!==-1){let y=m.subarray(0,g);a(_i(o,y)),p=Os(l,p,h,m.subarray(0,g+1));continue}try{i.entries.push({key:new TextDecoder("utf-8",{fatal:!0}).decode(_i(o,m))})}catch{}}}}var QS=!1;async function Bh(e,t,r,n){if(!wn(t.root.location))return await DD(e,t.root,t.rootHeight,pr,r,n)}async function DD(e,t,r,n,i,s){for(;;){let o=await Nc(e,t.location,s);if(QS&&console.log(t,r,o,n,i),Cc(o,r,n),!Ic(i,o.keyPrefix)){QS&&console.log("not found due to key prefix mismatch",i,o.keyPrefix);return}if(o.height===0)return BS(o.entries,i);let a=zS(o.entries,i);if(a===void 0)return;let{subtreeCommonPrefixLength:c}=a;i=i.subarray(c),t=a.node,n=a.key.subarray(c),--r}}async function ew(e,t,r){let{value:n}=t;if(n instanceof Uint8Array)return ls(n,r.byteRange);let{offset:i,length:s,dataFile:{baseUrl:o,relativePath:a}}=n,{store:c,path:u}=e.kvStoreContext.getKvStore(o);return await new tt(new Te(c,u+a),{offset:Number(i),length:Number(s)}).read(r)}function Ls(e){if(e===void 0)return"HEAD";if("generationNumber"in e)return`v${e.generationNumber}`;let{commitTime:t}=e;return Rc(t)}function Pc(e){if(e===void 0)return;let t=e.match(/^(?:v([1-9]\d*)|(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*)?Z))$/);if(t===null)throw new Error(`Invalid OCDBT version specifier: ${JSON.stringify(e)}`);let[,r]=t;if(r!==void 0){let n=BigInt(r);if(n>0xffffffffffffffffn)throw new Error(`Invalid generation number: ${n}`);return{generationNumber:n}}return{commitTime:PD(e)}}function PD(e){let t=e.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:(\.\d*))?Z$/);if(t===null)throw new Error(`Invalid commit timestamp: ${JSON.stringify(e)}`);let[,r,n]=t;return tw(Date.parse(r+"Z"),n)}function tw(e,t){let r=BigInt(e)*1000000n;if(t!==void 0&&t.length>1){let n=Number(t);r+=BigInt(Math.min(999999999,Math.round(n*1e9)))}return r}function Rc(e){let t=e%1000000000n,r=e/1000000000n;t<0n&&(t+=1000000000n,r-=1n);let n=new Date(Number(r)*1e3).toISOString();if(n.length!==24)throw new Error(`Invalid commit time: ${e} -> ${n}`);return n=n.slice(0,19),t!==0n&&(n+="."+t.toString().padStart(9,"0").replace(/0+$/,"")),n+="Z",n}var RD=new RegExp("^(\\d{0,4})(?:(?<=\\d{4})-(\\d{0,2})(?:(?<=\\d{2})-(\\d{0,2})(?:(?<=\\d{2})T(\\d{0,2})(?:(?<=\\d{2}):(\\d{0,2})(?:(?<=\\d{2}):(\\d{0,2})(?:(?<=\\d{2})(\\.\\d*)?(Z)?)?)?)?)?)?)?$");function Mi(e,t,r,n,i){t=t??"";let s=parseInt(t.padEnd(r,"0"),10),o=parseInt(t.padEnd(r,"9"),10);if(s>i)throw new Error(`Invalid ${e} prefix: ${t}`);return[Math.max(n,s),Math.min(i,o)]}function OD(e,t){let r=new Date(0);return r.setUTCFullYear(e),r.setUTCMonth(t),r.setUTCDate(0),r.getUTCDate()}function rw(e){let t=e.match(RD);if(t===null)throw new Error(`Expected prefix of ISO-8601 "YYYY-MM-DDThh:mm:ss.sssssssssZ" format, but received: ${JSON.stringify(e)}`);let r=Mi("year",t[1],4,0,9999),n=Mi("month",t[2],2,1,12),i=OD(r[1],n[1]),s=Mi("day",t[3],2,1,i),o=Mi("hour",t[4],2,0,23),a=Mi("minute",t[5],2,0,59),c=Mi("second",t[6],2,0,59),u=t[7]??".",l=t[8],f=u.padEnd(10,"0"),h=l===void 0?u.padEnd(10,"9"):f,p=[f,h];function d(m){let g=new Date(0);return g.setUTCFullYear(r[m]),g.setUTCMonth(n[m]-1),g.setUTCDate(s[m]),g.setUTCHours(o[m]),g.setUTCMinutes(a[m]),g.setUTCSeconds(c[m]),tw(g.getTime(),p[m])}return[d(0),d(1)]}async function nw(e,t,r,n){return e.chunkManager.memoize.get("ocdbt:version",()=>{let s=new qe(e.chunkManager.addRef(),{get:async({url:o,version:a},c)=>{let u=await Us(e,o,c),l=await UD(e,u,a,n);if(l===void 0)throw new Error(`Version ${Ls(a)} not found`);return{data:l.ref,size:0}},encodeKey:({url:o,version:a})=>{let c;return a!==void 0&&(c=Ls(a)),JSON.stringify([o,c])}});return s.registerDisposer(e.addRef()),s}).get({url:t,version:r},n)}async function UD(e,t,r,n){let{versionTree:i}=t;if(r===void 0){let{versionTreeNodes:a,inlineVersions:c}=i,u=c.length-1;return{ref:c[u],generationIndex:(a.at(-1)?.cumulativeNumGenerations??0n)+BigInt(u)}}let{ref:s,generationIndex:o}=await LD(e,t,r,n);if(s!==void 0)return{ref:s,generationIndex:o}}async function iw(e,t,r,n){let{generationIndex:i}=await FD(e,t,r,n);return i}async function sw(e,t,r,n){let{generationIndex:i}=await BD(e,t,r,n);return i}function zh(e){let{isInline:t,findInLeaf:r,findInInterior:n}=e;async function i(o,a,c,u){let{config:l,versionTree:f}=a,h=f.versionTreeNodes.at(-1)?.cumulativeNumGenerations??0n,{inlineVersions:p}=f;if(t(l,h,p,c)){let g=r(l,h,p,c);return{ref:p[g],generationIndex:h+BigInt(g)}}let{versionTreeNodes:d}=f;if(d.length===0)return{ref:void 0,generationIndex:0n};let m=n(l,0n,d,c);return m===void 0?{ref:void 0,generationIndex:0n}:await s(o,a.config,0n+m.cumulativeNumGenerations-m.numGenerations,m,c,u)}async function s(o,a,c,u,l,f){for(;;){let h=await Dc(o,u.location,f);if(kc(h,a,u.generationNumber,u.height,u.numGenerations),h.height===0){let d=h.entries,m=r(a,c,d,l);return{ref:d[m],generationIndex:c+BigInt(m)}}let p=n(a,c,h.entries,l);if(p===void 0)return{ref:void 0,generationIndex:c};u=p,c+=u.cumulativeNumGenerations-u.numGenerations}}return i}function $h(e,t,r){return"generationIndex"in r?r.generationIndex>=e:Lh(r,t[0])>=0}var LD=zh({isInline(e,t,r,n){return $h(t,r,n)},findInLeaf(e,t,r,n){return GS(t,r,n)},findInInterior(e,t,r,n){return jS(e.versionTreeArityLog2,t,r,n)}}),FD=zh({isInline(e,t,r,n){return $h(t,r,n)},findInLeaf(e,t,r,n){return Ai(t,r,n)},findInInterior(e,t,r,n){let i=Mc(e.versionTreeArityLog2,t,r,n);return r[i]}}),BD=zh({isInline(e,t,r,n){return $h(t,r,n)},findInLeaf(e,t,r,n){return Ai(t,r,n)},findInInterior(e,t,r,n){let i=Tc(t,r,n);return r[i]}});function ow(e,t){let{version:r,baseUrl:n}=e,i=r===void 0?"":`@${Ls(r)}/`;return n+`|ocdbt:${i}${Ze(t)}`}function aw(e,t){Rr(e);try{let r=(e.suffix??"").match(/^(?:@([^/]*)(?:\/|$))?(.*)$/),[,n,i]=r;return{baseUrl:t.store.getUrl(nn(t.path)),version:Pc(n),path:decodeURIComponent(i)}}catch(r){throw new Error(`Invalid URL: ${e.url}`,{cause:r})}}var Oc=class{constructor(t,r,n){this.sharedKvStoreContext=t,this.baseUrl=r,this.version=n}root;async getRoot(t){let{root:r}=this;return r===void 0&&(r=this.root=await nw(this.sharedKvStoreContext,this.baseUrl,this.version,t)),r}getUrl(t){return ow(this,t)}async stat(t,r){let n=await this.getRoot(r),i=new TextEncoder().encode(t),s=await Bh(this.sharedKvStoreContext,n,i,r);if(s===void 0)return;let{value:o}=s;return{totalSize:Number(o.length)}}async read(t,r){let n=await this.getRoot(r),i=new TextEncoder().encode(t),s=await Bh(this.sharedKvStoreContext,n,i,r);if(s!==void 0)return await ew(this.sharedKvStoreContext,s,r)}async list(t,r){let n=await this.getRoot(r),i=new TextEncoder().encode(t);return await XS(this.sharedKvStoreContext,n,i,r)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};var cw=!1;async function Vh(e,t,r){let{inclusiveMin:n,exclusiveMax:i}=r;cw&&console.log("listVersions",n,i);let s=n===void 0?{generationIndex:0n}:n,o=i===void 0?{generationIndex:t.versionTree.numGenerations}:i,{config:a,versionTree:c}=t,{versionTreeArityLog2:u}=a,l,f=[];{let m=c.versionTreeNodes.at(-1)?.cumulativeNumGenerations??0n;h(m,c.inlineVersions),await p(0n,c.versionTreeNodes)}function h(m,g){let y=Ai(m,g,s),I=Ai(m,g,o),_=m+BigInt(y);(l===void 0||_<l)&&(l=_);for(let E=y;E<I;++E)f.push(g[E])}async function p(m,g){r.signal?.throwIfAborted();let y=Mc(u,m,g,s),I=Tc(m,g,o);cw&&console.log("listVersions: visitInteriorEntries",s,o,`generationIndex=${m}`,`versionNodes.length=${g.length}`,y,I);let _=[];for(let E=y;E<I;++E){let b=g[E];_.push(d(m+b.cumulativeNumGenerations-b.numGenerations,b))}await Promise.all(_)}async function d(m,g){let y=await Dc(e,g.location,r);kc(y,a,g.generationNumber,g.height,g.numGenerations),y.height===0?h(m,y.entries):await p(m,y.entries)}return f.sort((m,g)=>mr(m.generationNumber,g.generationNumber)),{generationIndex:l??0n,versions:f}}async function zD(e,t,r,n,i,s){if(n<=r+i){let{versions:c}=await Vh(e,t,{inclusiveMin:{generationIndex:r},exclusiveMax:{generationIndex:n},...s});return c}let[{versions:o},{versions:a}]=await Promise.all([r,n-i/2n].map(c=>Vh(e,t,{inclusiveMin:{generationIndex:c},exclusiveMax:{generationIndex:c+i/2n},...s})));return[...o,...a]}async function uw(e,t){let{url:r}=t,n=r.suffix??"";if(n==="")return{offset:0,completions:[{value:"@",description:"Version specifier"}]};let i=n.match(/^@([^/]*)((?:\/|$).*)/);if(i===null)return;let[,s,o]=i;if(o!==""){Pc(s);return}let{base:a}=t,c=a.store.getUrl(nn(a.path));if(!s.startsWith("v")){let[u,l]=rw(s),f={signal:t.signal,progressListener:t.progressListener},h=await Us(e,c,f),[p,d]=await Promise.all([iw(e,h,{commitTime:u},f),sw(e,h,{commitTime:l+1n},f)]),g=(await zD(e,h,p,d,100n,{signal:t.signal,progressListener:t.progressListener})).map(y=>({value:`${Rc(y.commitTime)}/`,description:`v${y.generationNumber}`}));return g.reverse(),{offset:1,completions:g}}if(s==="v"){let{base:u}=t,f=(await Us(e,u.store.getUrl(u.path),t)).versionTree.inlineVersions.map(h=>({value:`v${h.generationNumber}/`,description:Rc(h.commitTime)}));return f.reverse(),{offset:1,completions:f}}return{offset:1,completions:[{value:`${s}/`}]}}function $D(e){return{scheme:"ocdbt",description:"OCDBT database",getKvStore(t,r){let{baseUrl:n,version:i,path:s}=aw(t,r);return{store:new Oc(e,n,i),path:s}},completeUrl(t){return uw(e,t)}}}wt.registerKvStoreAdapterProvider($D);var VD=["http://doc.s3.amazonaws.com/2006-03-01/","http://s3.amazonaws.com/doc/2006-03-01/"];function GD(e){return VD.includes(e.namespaceURI)&&e.tagName==="ListBucketResult"}async function Uc(e,t,r,n){let i="/";try{let s=await r(`${e}?list-type=2&prefix=${encodeURIComponent(t)}&delimiter=${encodeURIComponent(i)}&encoding-type=url`,{headers:{accept:"application/xml,text/xml"},signal:n.signal,progressListener:n.progressListener}),o=s.headers.get("content-type");if(o===null||/\b(application\/xml|text\/xml|text\/html)\b/i.exec(o)===null)throw new Error(`Expected XML content-type but received: ${o}`);let a=await s.text(),c=new DOMParser().parseFromString(a,"application/xml"),{documentElement:u}=c;if(!GD(u))throw new Error(`Received unexpected XML root element <${u.tagName} xmlns="${u.namespaceURI}">`);let l=u.namespaceURI,f=()=>l,h=c.evaluate("//CommonPrefixes/Prefix",c,f,XPathResult.UNORDERED_NODE_SNAPSHOT_TYPE,null),p=[];for(let g=0,y=h.snapshotLength;g<y;++g){let I=h.snapshotItem(g).textContent;I!==null&&(I=decodeURIComponent(I),p.push(I.substring(0,I.length-i.length)))}let d=[],m=c.evaluate("//Contents/Key",c,f,XPathResult.UNORDERED_NODE_SNAPSHOT_TYPE,null);for(let g=0,y=m.snapshotLength;g<y;++g){let I=m.snapshotItem(g).textContent;I!==null&&d.push({key:decodeURIComponent(I)})}return{directories:p,entries:d}}catch(s){throw new Error("S3-compatible listing not supported",{cause:s})}}function lw(e,t,r){let{baseUrl:n,path:i}=Dn(e);return Uc(n,i,t,r)}function jD(e){let t=new URL(e),r=t.pathname.match(/^\/([^/]+)(?:\/(.*))$/);if(r===null)return;let[,n,i]=r;return{bucketUrl:`${t.origin}/${n}/${t.search}`,bucket:decodeURIComponent(n),prefix:decodeURIComponent(i)}}async function fw(e,t,r){let n=jD(e);if(n===void 0)throw new Error(`Path-style S3 URL ${JSON.stringify(e)} must specify bucket`);let{bucketUrl:i,bucket:s,prefix:o}=n,a=await Uc(i,o,t,r),c=Ze(s)+"/";return{entries:a.entries.map(u=>({key:c+u.key})),directories:a.directories.map(u=>c+u)}}function qD(e){return e.getUncounted("s3:urlkind",()=>new Map)}async function hw(e,t,r,n,i){let s=qD(r),o=s.get(t);if(o==="virtual")return await lw(e,n,i);if(o==="path")return await fw(e,n,i);if(o!==null)try{let{result:a,urlKind:c}=await Promise.any([lw(e,n,i).then(u=>({result:u,urlKind:"virtual"})),fw(e,n,i).then(u=>({result:u,urlKind:"path"}))]);return s.set(t,c),a}catch(a){throw i.signal?.throwIfAborted(),s.set(t,null),new Error("Neither virtual hosted nor path-style S3 listing supported",{cause:a})}throw new Error("Neither virtual hosted nor path-style S3 listing supported")}var pw=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),dw=e=>{throw TypeError(e)},KD=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&dw("Object expected");var n,i;r&&(n=t[pw("asyncDispose")]),n===void 0&&(n=t[pw("dispose")],r&&(i=n)),typeof n!="function"&&dw("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},YD=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,u){return u=Error(c),u.name="SuppressedError",u.error=o,u.suppressed=a,u},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};var Lc=class{constructor(t,r,n,i,s=Re){this.sharedKvStoreContext=t,this.baseUrl=r,this.baseUrlForDisplay=n,this.knownToBeVirtualHostedStyle=i,this.fetchOkImpl=s}stat(t,r){let n=Ht(this.baseUrl,t);return pn(this,t,n,r,this.fetchOkImpl)}read(t,r){let n=Ht(this.baseUrl,t);return ei(this,t,n,r,this.fetchOkImpl)}list(t,r){var n=[];try{let{progressListener:o}=r,a=KD(n,o===void 0?void 0:new He(o,{message:`Listing prefix ${this.getUrl(t)}`}));return this.knownToBeVirtualHostedStyle?Uc(this.baseUrl,t,this.fetchOkImpl,r):hw(Ht(this.baseUrl,t),this.baseUrlForDisplay,this.sharedKvStoreContext.chunkManager.memoize,this.fetchOkImpl,r)}catch(o){var i=o,s=!0}finally{YD(n,i,s)}}getUrl(t){return Ht(this.baseUrlForDisplay,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function JD(e,t){return{scheme:"s3",description:"S3 (anonymous)",getKvStore(r){let n=(r.suffix??"").match(/^\/\/([^/]+)(\/.*)?$/);if(n===null)throw new Error("Invalid URL, expected `s3://<bucket>/<path>`");let[,i,s]=n;return{store:new t(e,`https://${i}.s3.amazonaws.com/`,`s3://${i}/`,!0),path:decodeURIComponent((s??"").substring(1))}}}}function HD(e,t,r){return{scheme:`s3+${t}`,description:`S3-compatible ${t} server`,getKvStore(n){let i=(n.suffix??"").match(/^\/\/([^/]+)(\/.*)?$/);if(i===null)throw new Error("Invalid URL, expected `s3+${httpScheme}://<host>/<path>`");let[,s,o]=i;return{store:new r(e,`${t}://${s}/`,`s3+${t}://${s}/`,!1),path:decodeURIComponent((o??"").substring(1))}}}}function mw(e,t){e.registerBaseKvStoreProvider(r=>JD(r,t));for(let r of["http","https"])e.registerBaseKvStoreProvider(n=>HD(n,r,t))}var Fc=class extends Lc{list(t,r){return ac(this.sharedKvStoreContext,this.getUrl(t),r)}};mw(wt,Fc);var vw=Ri(yw(),1);var jh=22,WD=65535,XD=101010256,ZD=101075792;function QD(e){let t=0,r;return async function(i,s,o){if(r!==void 0&&i>t&&i+s<=t+r.length)return r.subarray(i-t,i+s-t);let a=await e(i,s,o);return t=i,r=a,a}}function e3(e){let t=new DataView(e.buffer,e.byteOffset,e.byteLength),r=e.length;for(let n=r-jh;n>=0;--n){if(t.getUint32(n,!0)!==XD)continue;let i=t.getUint16(n+20,!0),s=r-n-jh;if(i!==s)continue;let o=t.getUint16(n+4,!0),a=t.getUint16(n+10,!0),c=t.getUint32(n+12,!0),u=t.getUint32(n+16,!0);return{eocdrOffset:n,diskNumber:o,entryCount:a,centralDirectorySize:c,centralDirectoryOffset:u}}}async function t3(e,t,r){let n=Math.min(jh+WD,t),i=t-n,s=await e(i,n,r),o=e3(s);if(o===void 0)throw new Error("End of central directory record signature not found; either not a zip file or file is truncated.");let{eocdrOffset:a,diskNumber:c,entryCount:u,centralDirectorySize:l,centralDirectoryOffset:f}=o;if(c!==0)throw new Error(`Multi-volume zip files are not supported. This is volume: ${c}`);let h=s.slice(a+22,s.length);return u===65535||f===4294967295?await n3(e,a,h,r):await xw(e,f,l,u,h,r)}var r3=117853008;async function n3(e,t,r,n){let i=t-20,s=await e(i,20,n),o=new DataView(s.buffer,s.byteOffset,s.byteLength);if(o.getUint32(0,!0)!==r3)throw new Error("invalid zip64 end of central directory locator signature");let a=o.getBigUint64(8,!0),c=await e(Number(a),56,n),u=new DataView(c.buffer,c.byteOffset,c.byteLength);if(u.getUint32(0,!0)!==ZD)throw new Error("invalid zip64 end of central directory record signature");let l=u.getBigUint64(32,!0),f=u.getBigUint64(40,!0),h=u.getBigUint64(48,!0);return xw(e,Number(h),Number(f),Number(l),r,n)}var i3=33639248;async function xw(e,t,r,n,i,s){let o=0,a=await e(t,r,s),c=[],u=new DataView(a.buffer,a.byteOffset,a.byteLength),l=new TextDecoder;for(let f=0;f<n;++f){let h=u.getUint32(o+0,!0);if(h!==i3)throw new Error(`invalid central directory file header signature: 0x${h.toString(16)}`);let p=u.getUint16(o+4,!0),d=u.getUint16(o+6,!0),m=u.getUint16(o+8,!0),g=u.getUint16(o+10,!0),y=u.getUint16(o+12,!0),I=u.getUint16(o+14,!0),_=u.getUint32(o+16,!0),E=u.getUint32(o+20,!0),b=u.getUint32(o+24,!0),C=u.getUint16(o+28,!0),v=u.getUint16(o+30,!0),w=u.getUint16(o+32,!0),x=u.getUint16(o+36,!0),T=u.getUint32(o+38,!0),M=u.getUint32(o+42,!0);if(m&64)throw new Error("strong encryption is not supported");o+=46;let P=a.subarray(o,o+=C),F=(m&2048)!==0,S=[];for(let L=0;L<v-3;){let j=u.getUint16(o+L+0,!0),V=u.getUint16(o+L+2,!0),Y=L+4,J=Y+V;if(J>v)throw new Error("extra field length exceeds extra field buffer size");S.push({id:j,offset:o+Y,length:V}),L=J}o+=v;let O=a.slice(o,o+=w);if(b===4294967295||E===4294967295||M===4294967295){let L=S.find(J=>J.id===1);if(L===void 0)throw new Error("expected zip64 extended information extra field");let{offset:j,length:V}=L,Y=0;if(b===4294967295){if(Y+8>V)throw new Error("zip64 extended information extra field does not include uncompressed size");b=Number(u.getBigUint64(j+Y,!0)),Y+=8}if(E===4294967295){if(Y+8>V)throw new Error("zip64 extended information extra field does not include compressed size");E=Number(u.getBigUint64(j+Y,!0)),Y+=8}if(M===4294967295){if(Y+8>V)throw new Error("zip64 extended information extra field does not include relative header offset");M=Number(u.getBigUint64(j+Y,!0)),Y+=8}}let R=S.find(L=>L.id===28789&&L.length>=6&&a[L.offset]===1&&u.getInt32(L.offset+1,!0)===(0,vw.buf)(P));if(R&&(P=a.slice(R.offset+5,R.offset+R.length),F=!0),g===0){let L=b;if((m&1)!==0&&(L+=12),E!==L)throw new Error(`compressed/uncompressed size mismatch for stored file: ${E} != ${L}`)}let N=l.decode(P);N=N.replaceAll("\\","/");let U={versionMadeBy:p,versionNeededToExtract:d,generalPurposeBitFlag:m,compressionMethod:g,lastModFileTime:y,lastModFileDate:I,crc32:_,compressedSize:E,uncompressedSize:b,nameBytes:P,commentBytes:O,internalFileAttributes:x,externalFileAttributes:T,relativeOffsetOfLocalHeader:M,fileName:N};c.push(U)}return{commentBytes:i,entries:c,sizeEstimate:i.length+a.length*2}}async function Sw(e,t,r){if(t.generalPurposeBitFlag&1)throw new Error("encrypted entries not supported");let n=await e(t.relativeOffsetOfLocalHeader,30,r),i=new DataView(n.buffer,n.byteOffset,n.byteLength),s=i.getUint32(0,!0);if(s!==67324752)throw new Error(`invalid local file header signature: 0x${s.toString(16)}`);let o=i.getUint16(26,!0),a=i.getUint16(28,!0);return t.relativeOffsetOfLocalHeader+n.length+o+a}async function ww(e,t,r){return await t3(QD(e),t,r)}var Bc=(e=>(e[e.STORE=0]="STORE",e[e.DEFLATE=8]="DEFLATE",e))(Bc||{});var Ew=(e,t)=>(t=Symbol[e])?t:Symbol.for("Symbol."+e),bw=e=>{throw TypeError(e)},s3=(e,t,r)=>{if(t!=null){typeof t!="object"&&typeof t!="function"&&bw("Object expected");var n,i;r&&(n=t[Ew("asyncDispose")]),n===void 0&&(n=t[Ew("dispose")],r&&(i=n)),typeof n!="function"&&bw("Object not disposable"),i&&(n=function(){try{i.call(this)}catch(s){return Promise.reject(s)}}),e.push([r,n,t])}else r&&e.push([r]);return t},o3=(e,t,r)=>{var n=typeof SuppressedError=="function"?SuppressedError:function(o,a,c,u){return u=Error(c),u.name="SuppressedError",u.error=o,u.suppressed=a,u},i=o=>t=r?new n(o,t,"An error was suppressed during disposal"):(r=!0,o),s=o=>{for(;o=e.pop();)try{var a=o[1]&&o[1].call(o[2]);if(o[0])return Promise.resolve(a).then(s,c=>(i(c),s()))}catch(c){i(c)}if(r)throw t};return s()};function _w(e){return async(t,r,n)=>{let i=await Pr(e,{throwIfMissing:!0,byteRange:{offset:t,length:r},strictByteRange:!0,signal:n.signal,progressListener:n.progressListener});return new Uint8Array(await i.response.arrayBuffer())}}function a3(e,t){let r=t.getUrl();return Sv(e,`zipMetadata:${r}`,{get:async(n,i)=>{var s=[];try{let c=s3(s,new He(i.progressListener,{message:`Reading ZIP central directory from ${r}`})),u=await t.stat(i);if(u?.totalSize===void 0)throw new Error(`Failed to determine ZIP file size: ${r}`);let l=await ww(_w(t),u.totalSize,i);return Zm(l.entries,f=>!f.fileName.endsWith("/")),l.entries.sort((f,h)=>St(f.fileName,h.fileName)),{data:l,size:l.sizeEstimate}}catch(c){var o=c,a=!0}finally{o3(s,o,a)}}})}async function c3(e,t,r){let n=a3(e,t);try{return await n.get(void 0,r)}finally{n.dispose()}}function Iw(e,t){let{entries:r}=e,n=er(r,t,(i,s)=>St(i,s.fileName));if(!(n<0))return r[n]}function u3(e,t){let{entries:r}=e,n=we(0,r.length,a=>r[a].fileName>=t),i=we(Math.min(r.length,n+1),r.length,a=>!r[a].fileName.startsWith(t)),s=[],o=[];for(let a=n;a<i;){let c=r[a],u=c.fileName.indexOf("/",t.length);if(u===-1)s.push({key:c.fileName}),++a;else{o.push(c.fileName.substring(0,u));let l=c.fileName.substring(0,u+1);a=we(a+1,i,f=>!r[f].fileName.startsWith(l))}}return{entries:s,directories:o}}var zc=class{constructor(t,r){this.chunkManager=t,this.base=r}metadata;async getMetadata(t){let{metadata:r}=this;return r===void 0&&(r=this.metadata=await c3(this.chunkManager,this.base,t)),r}getUrl(t){return this.base.getUrl()+`|zip:${Ze(t)}`}async stat(t,r){let n=Iw(await this.getMetadata(r),t);if(n!==void 0)return{totalSize:n.uncompressedSize}}async read(t,r){let n=Iw(await this.getMetadata(r),t);if(n===void 0)return;let{fileDataStart:i}=n;i===void 0&&(i=n.fileDataStart=await Sw(_w(this.base),n,r));let s=new tt(this.base,{offset:i,length:n.compressedSize});switch(n.compressionMethod){case Bc.STORE:break;case Bc.DEFLATE:s=new jr(s,"deflate-raw");break;default:throw new Error(`Unsupported compression method: ${n.compressionMethod}`)}return s.read(r)}async list(t,r){let n=await this.getMetadata(r);return u3(n,t)}get supportsOffsetReads(){return!0}get supportsSuffixReads(){return!0}};function l3(e){return{scheme:"zip",description:"ZIP archive",getKvStore(t,r){return Rr(t),{store:new zc(e.chunkManager,new Te(r.store,r.path)),path:decodeURIComponent(t.suffix??"")}}}}wt.registerKvStoreAdapterProvider(l3);var Cw=new to(self,!1);Cw.sendReady();globalThis.rpc=Cw;
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
