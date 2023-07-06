import{a as G2,b as J3}from"./chunk-ON5OQYWL.js";var y3=G2((k0,k3)=>{"use strict";var O6="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED";k3.exports=O6});var B3=G2((y0,T3)=>{"use strict";var I6=y3();function A3(){}function P3(){}P3.resetWarningCache=A3;T3.exports=function(){function c(r,s,i,f,n,l){if(l!==I6){var o=new Error("Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types");throw o.name="Invariant Violation",o}}c.isRequired=c;function a(){return c}var e={array:c,bigint:c,bool:c,func:c,number:c,object:c,string:c,symbol:c,any:c,arrayOf:a,element:c,elementType:c,instanceOf:a,node:c,objectOf:a,oneOf:a,oneOfType:a,shape:a,exact:a,checkPropTypes:P3,resetWarningCache:A3};return e.PropTypes=e,e}});var R3=G2((T0,F3)=>{F3.exports=B3()();var A0,P0});import{OBSERVER_ACTOR_CLASS as t0,EVALUATOR_ACTOR_CLASS as m0,PlayObserver as H0,SimplePlay as $3}from"@cogment/cogment-verse";import{useCallback as N1,useState as b1}from"react";import{Button as Z3,createLookup as c4,FpsCounter as a4,KeyboardControlList as e4,serializePlayerAction as J,Space as r4,TEACHER_ACTOR_CLASS as s4,TEACHER_NOOP_ACTION as i4,useDocumentKeypressListener as f4,usePressedKeys as n4,useRealTimeUpdate as l4}from"@cogment/cogment-verse";import{jsx as _2,jsxs as S1}from"react/jsx-runtime";var Z=new r4({discrete:{n:6}}),Y=c4();Y.setAction([],J(Z,0));Y.setAction(["FIRE"],J(Z,1));Y.setAction(["UP"],J(Z,2));Y.setAction(["DOWN"],J(Z,3));Y.setAction(["RIGHT"],J(Z,4));Y.setAction(["LEFT"],J(Z,5));var w1=({sendAction:c,fps:a=40,actorClass:e,observation:r,...s})=>{let[i,f]=b1(!1),[n,l]=b1("left"),o=N1(()=>f(N=>!N),[f]);f4("p",o);let t=r?.gamePlayerName,m="opacity-90 py-2 rounded-full items-center text-white font-bold, px-5 text-base outline-none",v=`bg-green-500 ${m}`,V=`bg-orange-500 ${m}`,p=n4(),g=N1(N=>{if(p.size===0&&e===s4){c(i4);return}let M=[];p.has("ArrowLeft")?M.push("LEFT"):p.has("ArrowRight")?M.push("RIGHT"):p.has("ArrowDown")?M.push("DOWN"):p.has("ArrowUp")?M.push("UP"):p.has("Enter")&&M.push("FIRE");let d=Y.getAction(M);c(d)},[p,c,e]),{currentFps:u}=l4(g,a,i);return S1("div",{...s,children:[S1("div",{className:"flex flex-row py-4 gap-1",children:[_2(Z3,{className:"flex-1",onClick:o,children:i?"Resume":"Pause"}),_2(a4,{className:"flex-none w-fit",value:u})]}),_2(e4,{items:[["Left/Right Arrows","FIRE and MOVE UP/DOWN"],["Up/Down Arrows","MOVE UP/DOWN"],["Enter","Fire"],["p","Pause/Unpause"]]})]})};import{useCallback as j3,useEffect as g1,useState as f2}from"react";import{Button as e0,DType as r0,FpsCounter as s0,KeyboardControlList as i0,serializePlayerAction as q2,Space as f0,Switch as n0,useDocumentKeypressListener as Y3,useRealTimeUpdate as l0,WEB_ACTOR_NAME as o0}from"@cogment/cogment-verse";function k1(c,a){var e=Object.keys(c);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(c);a&&(r=r.filter(function(s){return Object.getOwnPropertyDescriptor(c,s).enumerable})),e.push.apply(e,r)}return e}function H(c){for(var a=1;a<arguments.length;a++){var e=arguments[a]!=null?arguments[a]:{};a%2?k1(Object(e),!0).forEach(function(r){S(c,r,e[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(c,Object.getOwnPropertyDescriptors(e)):k1(Object(e)).forEach(function(r){Object.defineProperty(c,r,Object.getOwnPropertyDescriptor(e,r))})}return c}function T2(c){"@babel/helpers - typeof";return T2=typeof Symbol=="function"&&typeof Symbol.iterator=="symbol"?function(a){return typeof a}:function(a){return a&&typeof Symbol=="function"&&a.constructor===Symbol&&a!==Symbol.prototype?"symbol":typeof a},T2(c)}function o4(c,a){if(!(c instanceof a))throw new TypeError("Cannot call a class as a function")}function y1(c,a){for(var e=0;e<a.length;e++){var r=a[e];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(c,r.key,r)}}function t4(c,a,e){return a&&y1(c.prototype,a),e&&y1(c,e),Object.defineProperty(c,"prototype",{writable:!1}),c}function S(c,a,e){return a in c?Object.defineProperty(c,a,{value:e,enumerable:!0,configurable:!0,writable:!0}):c[a]=e,c}function n1(c,a){return H4(c)||v4(c,a)||a3(c,a)||h4()}function M2(c){return m4(c)||z4(c)||a3(c)||V4()}function m4(c){if(Array.isArray(c))return K2(c)}function H4(c){if(Array.isArray(c))return c}function z4(c){if(typeof Symbol<"u"&&c[Symbol.iterator]!=null||c["@@iterator"]!=null)return Array.from(c)}function v4(c,a){var e=c==null?null:typeof Symbol<"u"&&c[Symbol.iterator]||c["@@iterator"];if(e!=null){var r=[],s=!0,i=!1,f,n;try{for(e=e.call(c);!(s=(f=e.next()).done)&&(r.push(f.value),!(a&&r.length===a));s=!0);}catch(l){i=!0,n=l}finally{try{!s&&e.return!=null&&e.return()}finally{if(i)throw n}}return r}}function a3(c,a){if(c){if(typeof c=="string")return K2(c,a);var e=Object.prototype.toString.call(c).slice(8,-1);if(e==="Object"&&c.constructor&&(e=c.constructor.name),e==="Map"||e==="Set")return Array.from(c);if(e==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(e))return K2(c,a)}}function K2(c,a){(a==null||a>c.length)&&(a=c.length);for(var e=0,r=new Array(a);e<a;e++)r[e]=c[e];return r}function V4(){throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function h4(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}var A1=function(){},l1={},e3={},r3=null,s3={mark:A1,measure:A1};try{typeof window<"u"&&(l1=window),typeof document<"u"&&(e3=document),typeof MutationObserver<"u"&&(r3=MutationObserver),typeof performance<"u"&&(s3=performance)}catch{}var M4=l1.navigator||{},P1=M4.userAgent,T1=P1===void 0?"":P1,q=l1,L=e3,B1=r3,d2=s3,p0=!!q.document,O=!!L.documentElement&&!!L.head&&typeof L.addEventListener=="function"&&typeof L.createElement=="function",i3=~T1.indexOf("MSIE")||~T1.indexOf("Trident/"),g2,x2,N2,b2,S2,D="___FONT_AWESOME___",Q2=16,f3="fa",n3="svg-inline--fa",K="data-fa-i2svg",J2="data-fa-pseudo-element",p4="data-fa-pseudo-element-pending",o1="data-prefix",t1="data-icon",F1="fontawesome-i2svg",u4="async",C4=["HTML","HEAD","STYLE","SCRIPT"],l3=function(){try{return!0}catch{return!1}}(),C="classic",x="sharp",m1=[C,x];function p2(c){return new Proxy(c,{get:function(e,r){return r in e?e[r]:e[C]}})}var H2=p2((g2={},S(g2,C,{fa:"solid",fas:"solid","fa-solid":"solid",far:"regular","fa-regular":"regular",fal:"light","fa-light":"light",fat:"thin","fa-thin":"thin",fad:"duotone","fa-duotone":"duotone",fab:"brands","fa-brands":"brands",fak:"kit","fa-kit":"kit"}),S(g2,x,{fa:"solid",fass:"solid","fa-solid":"solid",fasr:"regular","fa-regular":"regular",fasl:"light","fa-light":"light"}),g2)),z2=p2((x2={},S(x2,C,{solid:"fas",regular:"far",light:"fal",thin:"fat",duotone:"fad",brands:"fab",kit:"fak"}),S(x2,x,{solid:"fass",regular:"fasr",light:"fasl"}),x2)),v2=p2((N2={},S(N2,C,{fab:"fa-brands",fad:"fa-duotone",fak:"fa-kit",fal:"fa-light",far:"fa-regular",fas:"fa-solid",fat:"fa-thin"}),S(N2,x,{fass:"fa-solid",fasr:"fa-regular",fasl:"fa-light"}),N2)),L4=p2((b2={},S(b2,C,{"fa-brands":"fab","fa-duotone":"fad","fa-kit":"fak","fa-light":"fal","fa-regular":"far","fa-solid":"fas","fa-thin":"fat"}),S(b2,x,{"fa-solid":"fass","fa-regular":"fasr","fa-light":"fasl"}),b2)),d4=/fa(s|r|l|t|d|b|k|ss|sr|sl)?[\-\ ]/,o3="fa-layers-text",g4=/Font ?Awesome ?([56 ]*)(Solid|Regular|Light|Thin|Duotone|Brands|Free|Pro|Sharp|Kit)?.*/i,x4=p2((S2={},S(S2,C,{900:"fas",400:"far",normal:"far",300:"fal",100:"fat"}),S(S2,x,{900:"fass",400:"fasr",300:"fasl"}),S2)),t3=[1,2,3,4,5,6,7,8,9,10],N4=t3.concat([11,12,13,14,15,16,17,18,19,20]),b4=["class","data-prefix","data-icon","data-fa-transform","data-fa-mask"],X={GROUP:"duotone-group",SWAP_OPACITY:"swap-opacity",PRIMARY:"primary",SECONDARY:"secondary"},V2=new Set;Object.keys(z2[C]).map(V2.add.bind(V2));Object.keys(z2[x]).map(V2.add.bind(V2));var S4=[].concat(m1,M2(V2),["2xs","xs","sm","lg","xl","2xl","beat","border","fade","beat-fade","bounce","flip-both","flip-horizontal","flip-vertical","flip","fw","inverse","layers-counter","layers-text","layers","li","pull-left","pull-right","pulse","rotate-180","rotate-270","rotate-90","rotate-by","shake","spin-pulse","spin-reverse","spin","stack-1x","stack-2x","stack","ul",X.GROUP,X.SWAP_OPACITY,X.PRIMARY,X.SECONDARY]).concat(t3.map(function(c){return"".concat(c,"x")})).concat(N4.map(function(c){return"w-".concat(c)})),t2=q.FontAwesomeConfig||{};function w4(c){var a=L.querySelector("script["+c+"]");if(a)return a.getAttribute(c)}function k4(c){return c===""?!0:c==="false"?!1:c==="true"?!0:c}L&&typeof L.querySelector=="function"&&(R1=[["data-family-prefix","familyPrefix"],["data-css-prefix","cssPrefix"],["data-family-default","familyDefault"],["data-style-default","styleDefault"],["data-replacement-class","replacementClass"],["data-auto-replace-svg","autoReplaceSvg"],["data-auto-add-css","autoAddCss"],["data-auto-a11y","autoA11y"],["data-search-pseudo-elements","searchPseudoElements"],["data-observe-mutations","observeMutations"],["data-mutate-approach","mutateApproach"],["data-keep-original-source","keepOriginalSource"],["data-measure-performance","measurePerformance"],["data-show-missing-icons","showMissingIcons"]],R1.forEach(function(c){var a=n1(c,2),e=a[0],r=a[1],s=k4(w4(e));s!=null&&(t2[r]=s)}));var R1,m3={styleDefault:"solid",familyDefault:"classic",cssPrefix:f3,replacementClass:n3,autoReplaceSvg:!0,autoAddCss:!0,autoA11y:!0,searchPseudoElements:!1,observeMutations:!0,mutateApproach:"async",keepOriginalSource:!0,measurePerformance:!1,showMissingIcons:!0};t2.familyPrefix&&(t2.cssPrefix=t2.familyPrefix);var r2=H(H({},m3),t2);r2.autoReplaceSvg||(r2.observeMutations=!1);var z={};Object.keys(m3).forEach(function(c){Object.defineProperty(z,c,{enumerable:!0,set:function(e){r2[c]=e,m2.forEach(function(r){return r(z)})},get:function(){return r2[c]}})});Object.defineProperty(z,"familyPrefix",{enumerable:!0,set:function(a){r2.cssPrefix=a,m2.forEach(function(e){return e(z)})},get:function(){return r2.cssPrefix}});q.FontAwesomeConfig=z;var m2=[];function y4(c){return m2.push(c),function(){m2.splice(m2.indexOf(c),1)}}var I=Q2,F={size:16,x:0,y:0,rotate:0,flipX:!1,flipY:!1};function A4(c){if(!(!c||!O)){var a=L.createElement("style");a.setAttribute("type","text/css"),a.innerHTML=c;for(var e=L.head.childNodes,r=null,s=e.length-1;s>-1;s--){var i=e[s],f=(i.tagName||"").toUpperCase();["STYLE","LINK"].indexOf(f)>-1&&(r=i)}return L.head.insertBefore(a,r),c}}var P4="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";function h2(){for(var c=12,a="";c-- >0;)a+=P4[Math.random()*62|0];return a}function s2(c){for(var a=[],e=(c||[]).length>>>0;e--;)a[e]=c[e];return a}function H1(c){return c.classList?s2(c.classList):(c.getAttribute("class")||"").split(" ").filter(function(a){return a})}function H3(c){return"".concat(c).replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/'/g,"&#39;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function T4(c){return Object.keys(c||{}).reduce(function(a,e){return a+"".concat(e,'="').concat(H3(c[e]),'" ')},"").trim()}function R2(c){return Object.keys(c||{}).reduce(function(a,e){return a+"".concat(e,": ").concat(c[e].trim(),";")},"")}function z1(c){return c.size!==F.size||c.x!==F.x||c.y!==F.y||c.rotate!==F.rotate||c.flipX||c.flipY}function B4(c){var a=c.transform,e=c.containerWidth,r=c.iconWidth,s={transform:"translate(".concat(e/2," 256)")},i="translate(".concat(a.x*32,", ").concat(a.y*32,") "),f="scale(".concat(a.size/16*(a.flipX?-1:1),", ").concat(a.size/16*(a.flipY?-1:1),") "),n="rotate(".concat(a.rotate," 0 0)"),l={transform:"".concat(i," ").concat(f," ").concat(n)},o={transform:"translate(".concat(r/2*-1," -256)")};return{outer:s,inner:l,path:o}}function F4(c){var a=c.transform,e=c.width,r=e===void 0?Q2:e,s=c.height,i=s===void 0?Q2:s,f=c.startCentered,n=f===void 0?!1:f,l="";return n&&i3?l+="translate(".concat(a.x/I-r/2,"em, ").concat(a.y/I-i/2,"em) "):n?l+="translate(calc(-50% + ".concat(a.x/I,"em), calc(-50% + ").concat(a.y/I,"em)) "):l+="translate(".concat(a.x/I,"em, ").concat(a.y/I,"em) "),l+="scale(".concat(a.size/I*(a.flipX?-1:1),", ").concat(a.size/I*(a.flipY?-1:1),") "),l+="rotate(".concat(a.rotate,"deg) "),l}var R4=`:root, :host {
  --fa-font-solid: normal 900 1em/1 "Font Awesome 6 Solid";
  --fa-font-regular: normal 400 1em/1 "Font Awesome 6 Regular";
  --fa-font-light: normal 300 1em/1 "Font Awesome 6 Light";
  --fa-font-thin: normal 100 1em/1 "Font Awesome 6 Thin";
  --fa-font-duotone: normal 900 1em/1 "Font Awesome 6 Duotone";
  --fa-font-sharp-solid: normal 900 1em/1 "Font Awesome 6 Sharp";
  --fa-font-sharp-regular: normal 400 1em/1 "Font Awesome 6 Sharp";
  --fa-font-sharp-light: normal 300 1em/1 "Font Awesome 6 Sharp";
  --fa-font-brands: normal 400 1em/1 "Font Awesome 6 Brands";
}

svg:not(:root).svg-inline--fa, svg:not(:host).svg-inline--fa {
  overflow: visible;
  box-sizing: content-box;
}

.svg-inline--fa {
  display: var(--fa-display, inline-block);
  height: 1em;
  overflow: visible;
  vertical-align: -0.125em;
}
.svg-inline--fa.fa-2xs {
  vertical-align: 0.1em;
}
.svg-inline--fa.fa-xs {
  vertical-align: 0em;
}
.svg-inline--fa.fa-sm {
  vertical-align: -0.0714285705em;
}
.svg-inline--fa.fa-lg {
  vertical-align: -0.2em;
}
.svg-inline--fa.fa-xl {
  vertical-align: -0.25em;
}
.svg-inline--fa.fa-2xl {
  vertical-align: -0.3125em;
}
.svg-inline--fa.fa-pull-left {
  margin-right: var(--fa-pull-margin, 0.3em);
  width: auto;
}
.svg-inline--fa.fa-pull-right {
  margin-left: var(--fa-pull-margin, 0.3em);
  width: auto;
}
.svg-inline--fa.fa-li {
  width: var(--fa-li-width, 2em);
  top: 0.25em;
}
.svg-inline--fa.fa-fw {
  width: var(--fa-fw-width, 1.25em);
}

.fa-layers svg.svg-inline--fa {
  bottom: 0;
  left: 0;
  margin: auto;
  position: absolute;
  right: 0;
  top: 0;
}

.fa-layers-counter, .fa-layers-text {
  display: inline-block;
  position: absolute;
  text-align: center;
}

.fa-layers {
  display: inline-block;
  height: 1em;
  position: relative;
  text-align: center;
  vertical-align: -0.125em;
  width: 1em;
}
.fa-layers svg.svg-inline--fa {
  -webkit-transform-origin: center center;
          transform-origin: center center;
}

.fa-layers-text {
  left: 50%;
  top: 50%;
  -webkit-transform: translate(-50%, -50%);
          transform: translate(-50%, -50%);
  -webkit-transform-origin: center center;
          transform-origin: center center;
}

.fa-layers-counter {
  background-color: var(--fa-counter-background-color, #ff253a);
  border-radius: var(--fa-counter-border-radius, 1em);
  box-sizing: border-box;
  color: var(--fa-inverse, #fff);
  line-height: var(--fa-counter-line-height, 1);
  max-width: var(--fa-counter-max-width, 5em);
  min-width: var(--fa-counter-min-width, 1.5em);
  overflow: hidden;
  padding: var(--fa-counter-padding, 0.25em 0.5em);
  right: var(--fa-right, 0);
  text-overflow: ellipsis;
  top: var(--fa-top, 0);
  -webkit-transform: scale(var(--fa-counter-scale, 0.25));
          transform: scale(var(--fa-counter-scale, 0.25));
  -webkit-transform-origin: top right;
          transform-origin: top right;
}

.fa-layers-bottom-right {
  bottom: var(--fa-bottom, 0);
  right: var(--fa-right, 0);
  top: auto;
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: bottom right;
          transform-origin: bottom right;
}

.fa-layers-bottom-left {
  bottom: var(--fa-bottom, 0);
  left: var(--fa-left, 0);
  right: auto;
  top: auto;
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: bottom left;
          transform-origin: bottom left;
}

.fa-layers-top-right {
  top: var(--fa-top, 0);
  right: var(--fa-right, 0);
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: top right;
          transform-origin: top right;
}

.fa-layers-top-left {
  left: var(--fa-left, 0);
  right: auto;
  top: var(--fa-top, 0);
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: top left;
          transform-origin: top left;
}

.fa-1x {
  font-size: 1em;
}

.fa-2x {
  font-size: 2em;
}

.fa-3x {
  font-size: 3em;
}

.fa-4x {
  font-size: 4em;
}

.fa-5x {
  font-size: 5em;
}

.fa-6x {
  font-size: 6em;
}

.fa-7x {
  font-size: 7em;
}

.fa-8x {
  font-size: 8em;
}

.fa-9x {
  font-size: 9em;
}

.fa-10x {
  font-size: 10em;
}

.fa-2xs {
  font-size: 0.625em;
  line-height: 0.1em;
  vertical-align: 0.225em;
}

.fa-xs {
  font-size: 0.75em;
  line-height: 0.0833333337em;
  vertical-align: 0.125em;
}

.fa-sm {
  font-size: 0.875em;
  line-height: 0.0714285718em;
  vertical-align: 0.0535714295em;
}

.fa-lg {
  font-size: 1.25em;
  line-height: 0.05em;
  vertical-align: -0.075em;
}

.fa-xl {
  font-size: 1.5em;
  line-height: 0.0416666682em;
  vertical-align: -0.125em;
}

.fa-2xl {
  font-size: 2em;
  line-height: 0.03125em;
  vertical-align: -0.1875em;
}

.fa-fw {
  text-align: center;
  width: 1.25em;
}

.fa-ul {
  list-style-type: none;
  margin-left: var(--fa-li-margin, 2.5em);
  padding-left: 0;
}
.fa-ul > li {
  position: relative;
}

.fa-li {
  left: calc(var(--fa-li-width, 2em) * -1);
  position: absolute;
  text-align: center;
  width: var(--fa-li-width, 2em);
  line-height: inherit;
}

.fa-border {
  border-color: var(--fa-border-color, #eee);
  border-radius: var(--fa-border-radius, 0.1em);
  border-style: var(--fa-border-style, solid);
  border-width: var(--fa-border-width, 0.08em);
  padding: var(--fa-border-padding, 0.2em 0.25em 0.15em);
}

.fa-pull-left {
  float: left;
  margin-right: var(--fa-pull-margin, 0.3em);
}

.fa-pull-right {
  float: right;
  margin-left: var(--fa-pull-margin, 0.3em);
}

.fa-beat {
  -webkit-animation-name: fa-beat;
          animation-name: fa-beat;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, ease-in-out);
          animation-timing-function: var(--fa-animation-timing, ease-in-out);
}

.fa-bounce {
  -webkit-animation-name: fa-bounce;
          animation-name: fa-bounce;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.28, 0.84, 0.42, 1));
          animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.28, 0.84, 0.42, 1));
}

.fa-fade {
  -webkit-animation-name: fa-fade;
          animation-name: fa-fade;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
          animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
}

.fa-beat-fade {
  -webkit-animation-name: fa-beat-fade;
          animation-name: fa-beat-fade;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
          animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
}

.fa-flip {
  -webkit-animation-name: fa-flip;
          animation-name: fa-flip;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, ease-in-out);
          animation-timing-function: var(--fa-animation-timing, ease-in-out);
}

.fa-shake {
  -webkit-animation-name: fa-shake;
          animation-name: fa-shake;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, linear);
          animation-timing-function: var(--fa-animation-timing, linear);
}

.fa-spin {
  -webkit-animation-name: fa-spin;
          animation-name: fa-spin;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 2s);
          animation-duration: var(--fa-animation-duration, 2s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, linear);
          animation-timing-function: var(--fa-animation-timing, linear);
}

.fa-spin-reverse {
  --fa-animation-direction: reverse;
}

.fa-pulse,
.fa-spin-pulse {
  -webkit-animation-name: fa-spin;
          animation-name: fa-spin;
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, steps(8));
          animation-timing-function: var(--fa-animation-timing, steps(8));
}

@media (prefers-reduced-motion: reduce) {
  .fa-beat,
.fa-bounce,
.fa-fade,
.fa-beat-fade,
.fa-flip,
.fa-pulse,
.fa-shake,
.fa-spin,
.fa-spin-pulse {
    -webkit-animation-delay: -1ms;
            animation-delay: -1ms;
    -webkit-animation-duration: 1ms;
            animation-duration: 1ms;
    -webkit-animation-iteration-count: 1;
            animation-iteration-count: 1;
    -webkit-transition-delay: 0s;
            transition-delay: 0s;
    -webkit-transition-duration: 0s;
            transition-duration: 0s;
  }
}
@-webkit-keyframes fa-beat {
  0%, 90% {
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  45% {
    -webkit-transform: scale(var(--fa-beat-scale, 1.25));
            transform: scale(var(--fa-beat-scale, 1.25));
  }
}
@keyframes fa-beat {
  0%, 90% {
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  45% {
    -webkit-transform: scale(var(--fa-beat-scale, 1.25));
            transform: scale(var(--fa-beat-scale, 1.25));
  }
}
@-webkit-keyframes fa-bounce {
  0% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  10% {
    -webkit-transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
            transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
  }
  30% {
    -webkit-transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
            transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
  }
  50% {
    -webkit-transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
            transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
  }
  57% {
    -webkit-transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
            transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
  }
  64% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  100% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
}
@keyframes fa-bounce {
  0% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  10% {
    -webkit-transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
            transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
  }
  30% {
    -webkit-transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
            transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
  }
  50% {
    -webkit-transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
            transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
  }
  57% {
    -webkit-transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
            transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
  }
  64% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  100% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
}
@-webkit-keyframes fa-fade {
  50% {
    opacity: var(--fa-fade-opacity, 0.4);
  }
}
@keyframes fa-fade {
  50% {
    opacity: var(--fa-fade-opacity, 0.4);
  }
}
@-webkit-keyframes fa-beat-fade {
  0%, 100% {
    opacity: var(--fa-beat-fade-opacity, 0.4);
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  50% {
    opacity: 1;
    -webkit-transform: scale(var(--fa-beat-fade-scale, 1.125));
            transform: scale(var(--fa-beat-fade-scale, 1.125));
  }
}
@keyframes fa-beat-fade {
  0%, 100% {
    opacity: var(--fa-beat-fade-opacity, 0.4);
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  50% {
    opacity: 1;
    -webkit-transform: scale(var(--fa-beat-fade-scale, 1.125));
            transform: scale(var(--fa-beat-fade-scale, 1.125));
  }
}
@-webkit-keyframes fa-flip {
  50% {
    -webkit-transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
            transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
  }
}
@keyframes fa-flip {
  50% {
    -webkit-transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
            transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
  }
}
@-webkit-keyframes fa-shake {
  0% {
    -webkit-transform: rotate(-15deg);
            transform: rotate(-15deg);
  }
  4% {
    -webkit-transform: rotate(15deg);
            transform: rotate(15deg);
  }
  8%, 24% {
    -webkit-transform: rotate(-18deg);
            transform: rotate(-18deg);
  }
  12%, 28% {
    -webkit-transform: rotate(18deg);
            transform: rotate(18deg);
  }
  16% {
    -webkit-transform: rotate(-22deg);
            transform: rotate(-22deg);
  }
  20% {
    -webkit-transform: rotate(22deg);
            transform: rotate(22deg);
  }
  32% {
    -webkit-transform: rotate(-12deg);
            transform: rotate(-12deg);
  }
  36% {
    -webkit-transform: rotate(12deg);
            transform: rotate(12deg);
  }
  40%, 100% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
}
@keyframes fa-shake {
  0% {
    -webkit-transform: rotate(-15deg);
            transform: rotate(-15deg);
  }
  4% {
    -webkit-transform: rotate(15deg);
            transform: rotate(15deg);
  }
  8%, 24% {
    -webkit-transform: rotate(-18deg);
            transform: rotate(-18deg);
  }
  12%, 28% {
    -webkit-transform: rotate(18deg);
            transform: rotate(18deg);
  }
  16% {
    -webkit-transform: rotate(-22deg);
            transform: rotate(-22deg);
  }
  20% {
    -webkit-transform: rotate(22deg);
            transform: rotate(22deg);
  }
  32% {
    -webkit-transform: rotate(-12deg);
            transform: rotate(-12deg);
  }
  36% {
    -webkit-transform: rotate(12deg);
            transform: rotate(12deg);
  }
  40%, 100% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(360deg);
            transform: rotate(360deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(360deg);
            transform: rotate(360deg);
  }
}
.fa-rotate-90 {
  -webkit-transform: rotate(90deg);
          transform: rotate(90deg);
}

.fa-rotate-180 {
  -webkit-transform: rotate(180deg);
          transform: rotate(180deg);
}

.fa-rotate-270 {
  -webkit-transform: rotate(270deg);
          transform: rotate(270deg);
}

.fa-flip-horizontal {
  -webkit-transform: scale(-1, 1);
          transform: scale(-1, 1);
}

.fa-flip-vertical {
  -webkit-transform: scale(1, -1);
          transform: scale(1, -1);
}

.fa-flip-both,
.fa-flip-horizontal.fa-flip-vertical {
  -webkit-transform: scale(-1, -1);
          transform: scale(-1, -1);
}

.fa-rotate-by {
  -webkit-transform: rotate(var(--fa-rotate-angle, none));
          transform: rotate(var(--fa-rotate-angle, none));
}

.fa-stack {
  display: inline-block;
  vertical-align: middle;
  height: 2em;
  position: relative;
  width: 2.5em;
}

.fa-stack-1x,
.fa-stack-2x {
  bottom: 0;
  left: 0;
  margin: auto;
  position: absolute;
  right: 0;
  top: 0;
  z-index: var(--fa-stack-z-index, auto);
}

.svg-inline--fa.fa-stack-1x {
  height: 1em;
  width: 1.25em;
}
.svg-inline--fa.fa-stack-2x {
  height: 2em;
  width: 2.5em;
}

.fa-inverse {
  color: var(--fa-inverse, #fff);
}

.sr-only,
.fa-sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

.sr-only-focusable:not(:focus),
.fa-sr-only-focusable:not(:focus) {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

.svg-inline--fa .fa-primary {
  fill: var(--fa-primary-color, currentColor);
  opacity: var(--fa-primary-opacity, 1);
}

.svg-inline--fa .fa-secondary {
  fill: var(--fa-secondary-color, currentColor);
  opacity: var(--fa-secondary-opacity, 0.4);
}

.svg-inline--fa.fa-swap-opacity .fa-primary {
  opacity: var(--fa-secondary-opacity, 0.4);
}

.svg-inline--fa.fa-swap-opacity .fa-secondary {
  opacity: var(--fa-primary-opacity, 1);
}

.svg-inline--fa mask .fa-primary,
.svg-inline--fa mask .fa-secondary {
  fill: black;
}

.fad.fa-inverse,
.fa-duotone.fa-inverse {
  color: var(--fa-inverse, #fff);
}`;function z3(){var c=f3,a=n3,e=z.cssPrefix,r=z.replacementClass,s=R4;if(e!==c||r!==a){var i=new RegExp("\\.".concat(c,"\\-"),"g"),f=new RegExp("\\--".concat(c,"\\-"),"g"),n=new RegExp("\\.".concat(a),"g");s=s.replace(i,".".concat(e,"-")).replace(f,"--".concat(e,"-")).replace(n,".".concat(r))}return s}var D1=!1;function j2(){z.autoAddCss&&!D1&&(A4(z3()),D1=!0)}var D4={mixout:function(){return{dom:{css:z3,insertCss:j2}}},hooks:function(){return{beforeDOMElementCreation:function(){j2()},beforeI2svg:function(){j2()}}}},E=q||{};E[D]||(E[D]={});E[D].styles||(E[D].styles={});E[D].hooks||(E[D].hooks={});E[D].shims||(E[D].shims=[]);var P=E[D],v3=[],E4=function c(){L.removeEventListener("DOMContentLoaded",c),B2=1,v3.map(function(a){return a()})},B2=!1;O&&(B2=(L.documentElement.doScroll?/^loaded|^c/:/^loaded|^i|^c/).test(L.readyState),B2||L.addEventListener("DOMContentLoaded",E4));function U4(c){O&&(B2?setTimeout(c,0):v3.push(c))}function u2(c){var a=c.tag,e=c.attributes,r=e===void 0?{}:e,s=c.children,i=s===void 0?[]:s;return typeof c=="string"?H3(c):"<".concat(a," ").concat(T4(r),">").concat(i.map(u2).join(""),"</").concat(a,">")}function E1(c,a,e){if(c&&c[a]&&c[a][e])return{prefix:a,iconName:e,icon:c[a][e]}}var O4=function(a,e){return function(r,s,i,f){return a.call(e,r,s,i,f)}},Y2=function(a,e,r,s){var i=Object.keys(a),f=i.length,n=s!==void 0?O4(e,s):e,l,o,t;for(r===void 0?(l=1,t=a[i[0]]):(l=0,t=r);l<f;l++)o=i[l],t=n(t,a[o],o,a);return t};function I4(c){for(var a=[],e=0,r=c.length;e<r;){var s=c.charCodeAt(e++);if(s>=55296&&s<=56319&&e<r){var i=c.charCodeAt(e++);(i&64512)==56320?a.push(((s&1023)<<10)+(i&1023)+65536):(a.push(s),e--)}else a.push(s)}return a}function Z2(c){var a=I4(c);return a.length===1?a[0].toString(16):null}function q4(c,a){var e=c.length,r=c.charCodeAt(a),s;return r>=55296&&r<=56319&&e>a+1&&(s=c.charCodeAt(a+1),s>=56320&&s<=57343)?(r-55296)*1024+s-56320+65536:r}function U1(c){return Object.keys(c).reduce(function(a,e){var r=c[e],s=!!r.icon;return s?a[r.iconName]=r.icon:a[e]=r,a},{})}function c1(c,a){var e=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{},r=e.skipHooks,s=r===void 0?!1:r,i=U1(a);typeof P.hooks.addPack=="function"&&!s?P.hooks.addPack(c,U1(a)):P.styles[c]=H(H({},P.styles[c]||{}),i),c==="fas"&&c1("fa",a)}var w2,k2,y2,c2=P.styles,W4=P.shims,G4=(w2={},S(w2,C,Object.values(v2[C])),S(w2,x,Object.values(v2[x])),w2),v1=null,V3={},h3={},M3={},p3={},u3={},_4=(k2={},S(k2,C,Object.keys(H2[C])),S(k2,x,Object.keys(H2[x])),k2);function j4(c){return~S4.indexOf(c)}function Y4(c,a){var e=a.split("-"),r=e[0],s=e.slice(1).join("-");return r===c&&s!==""&&!j4(s)?s:null}var C3=function(){var a=function(i){return Y2(c2,function(f,n,l){return f[l]=Y2(n,i,{}),f},{})};V3=a(function(s,i,f){if(i[3]&&(s[i[3]]=f),i[2]){var n=i[2].filter(function(l){return typeof l=="number"});n.forEach(function(l){s[l.toString(16)]=f})}return s}),h3=a(function(s,i,f){if(s[f]=f,i[2]){var n=i[2].filter(function(l){return typeof l=="string"});n.forEach(function(l){s[l]=f})}return s}),u3=a(function(s,i,f){var n=i[2];return s[f]=f,n.forEach(function(l){s[l]=f}),s});var e="far"in c2||z.autoFetchSvg,r=Y2(W4,function(s,i){var f=i[0],n=i[1],l=i[2];return n==="far"&&!e&&(n="fas"),typeof f=="string"&&(s.names[f]={prefix:n,iconName:l}),typeof f=="number"&&(s.unicodes[f.toString(16)]={prefix:n,iconName:l}),s},{names:{},unicodes:{}});M3=r.names,p3=r.unicodes,v1=D2(z.styleDefault,{family:z.familyDefault})};y4(function(c){v1=D2(c.styleDefault,{family:z.familyDefault})});C3();function V1(c,a){return(V3[c]||{})[a]}function X4(c,a){return(h3[c]||{})[a]}function $(c,a){return(u3[c]||{})[a]}function L3(c){return M3[c]||{prefix:null,iconName:null}}function $4(c){var a=p3[c],e=V1("fas",c);return a||(e?{prefix:"fas",iconName:e}:null)||{prefix:null,iconName:null}}function W(){return v1}var h1=function(){return{prefix:null,iconName:null,rest:[]}};function D2(c){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},e=a.family,r=e===void 0?C:e,s=H2[r][c],i=z2[r][c]||z2[r][s],f=c in P.styles?c:null;return i||f||null}var O1=(y2={},S(y2,C,Object.keys(v2[C])),S(y2,x,Object.keys(v2[x])),y2);function E2(c){var a,e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},r=e.skipLookups,s=r===void 0?!1:r,i=(a={},S(a,C,"".concat(z.cssPrefix,"-").concat(C)),S(a,x,"".concat(z.cssPrefix,"-").concat(x)),a),f=null,n=C;(c.includes(i[C])||c.some(function(o){return O1[C].includes(o)}))&&(n=C),(c.includes(i[x])||c.some(function(o){return O1[x].includes(o)}))&&(n=x);var l=c.reduce(function(o,t){var m=Y4(z.cssPrefix,t);if(c2[t]?(t=G4[n].includes(t)?L4[n][t]:t,f=t,o.prefix=t):_4[n].indexOf(t)>-1?(f=t,o.prefix=D2(t,{family:n})):m?o.iconName=m:t!==z.replacementClass&&t!==i[C]&&t!==i[x]&&o.rest.push(t),!s&&o.prefix&&o.iconName){var v=f==="fa"?L3(o.iconName):{},V=$(o.prefix,o.iconName);v.prefix&&(f=null),o.iconName=v.iconName||V||o.iconName,o.prefix=v.prefix||o.prefix,o.prefix==="far"&&!c2.far&&c2.fas&&!z.autoFetchSvg&&(o.prefix="fas")}return o},h1());return(c.includes("fa-brands")||c.includes("fab"))&&(l.prefix="fab"),(c.includes("fa-duotone")||c.includes("fad"))&&(l.prefix="fad"),!l.prefix&&n===x&&(c2.fass||z.autoFetchSvg)&&(l.prefix="fass",l.iconName=$(l.prefix,l.iconName)||l.iconName),(l.prefix==="fa"||f==="fa")&&(l.prefix=W()||"fas"),l}var K4=function(){function c(){o4(this,c),this.definitions={}}return t4(c,[{key:"add",value:function(){for(var e=this,r=arguments.length,s=new Array(r),i=0;i<r;i++)s[i]=arguments[i];var f=s.reduce(this._pullDefinitions,{});Object.keys(f).forEach(function(n){e.definitions[n]=H(H({},e.definitions[n]||{}),f[n]),c1(n,f[n]);var l=v2[C][n];l&&c1(l,f[n]),C3()})}},{key:"reset",value:function(){this.definitions={}}},{key:"_pullDefinitions",value:function(e,r){var s=r.prefix&&r.iconName&&r.icon?{0:r}:r;return Object.keys(s).map(function(i){var f=s[i],n=f.prefix,l=f.iconName,o=f.icon,t=o[2];e[n]||(e[n]={}),t.length>0&&t.forEach(function(m){typeof m=="string"&&(e[n][m]=o)}),e[n][l]=o}),e}}]),c}(),I1=[],a2={},e2={},Q4=Object.keys(e2);function J4(c,a){var e=a.mixoutsTo;return I1=c,a2={},Object.keys(e2).forEach(function(r){Q4.indexOf(r)===-1&&delete e2[r]}),I1.forEach(function(r){var s=r.mixout?r.mixout():{};if(Object.keys(s).forEach(function(f){typeof s[f]=="function"&&(e[f]=s[f]),T2(s[f])==="object"&&Object.keys(s[f]).forEach(function(n){e[f]||(e[f]={}),e[f][n]=s[f][n]})}),r.hooks){var i=r.hooks();Object.keys(i).forEach(function(f){a2[f]||(a2[f]=[]),a2[f].push(i[f])})}r.provides&&r.provides(e2)}),e}function a1(c,a){for(var e=arguments.length,r=new Array(e>2?e-2:0),s=2;s<e;s++)r[s-2]=arguments[s];var i=a2[c]||[];return i.forEach(function(f){a=f.apply(null,[a].concat(r))}),a}function Q(c){for(var a=arguments.length,e=new Array(a>1?a-1:0),r=1;r<a;r++)e[r-1]=arguments[r];var s=a2[c]||[];s.forEach(function(i){i.apply(null,e)})}function U(){var c=arguments[0],a=Array.prototype.slice.call(arguments,1);return e2[c]?e2[c].apply(null,a):void 0}function e1(c){c.prefix==="fa"&&(c.prefix="fas");var a=c.iconName,e=c.prefix||W();if(a)return a=$(e,a)||a,E1(d3.definitions,e,a)||E1(P.styles,e,a)}var d3=new K4,Z4=function(){z.autoReplaceSvg=!1,z.observeMutations=!1,Q("noAuto")},c6={i2svg:function(){var a=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};return O?(Q("beforeI2svg",a),U("pseudoElements2svg",a),U("i2svg",a)):Promise.reject("Operation requires a DOM of some kind.")},watch:function(){var a=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},e=a.autoReplaceSvgRoot;z.autoReplaceSvg===!1&&(z.autoReplaceSvg=!0),z.observeMutations=!0,U4(function(){e6({autoReplaceSvgRoot:e}),Q("watch",a)})}},a6={icon:function(a){if(a===null)return null;if(T2(a)==="object"&&a.prefix&&a.iconName)return{prefix:a.prefix,iconName:$(a.prefix,a.iconName)||a.iconName};if(Array.isArray(a)&&a.length===2){var e=a[1].indexOf("fa-")===0?a[1].slice(3):a[1],r=D2(a[0]);return{prefix:r,iconName:$(r,e)||e}}if(typeof a=="string"&&(a.indexOf("".concat(z.cssPrefix,"-"))>-1||a.match(d4))){var s=E2(a.split(" "),{skipLookups:!0});return{prefix:s.prefix||W(),iconName:$(s.prefix,s.iconName)||s.iconName}}if(typeof a=="string"){var i=W();return{prefix:i,iconName:$(i,a)||a}}}},y={noAuto:Z4,config:z,dom:c6,parse:a6,library:d3,findIconDefinition:e1,toHtml:u2},e6=function(){var a=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},e=a.autoReplaceSvgRoot,r=e===void 0?L:e;(Object.keys(P.styles).length>0||z.autoFetchSvg)&&O&&z.autoReplaceSvg&&y.dom.i2svg({node:r})};function U2(c,a){return Object.defineProperty(c,"abstract",{get:a}),Object.defineProperty(c,"html",{get:function(){return c.abstract.map(function(r){return u2(r)})}}),Object.defineProperty(c,"node",{get:function(){if(O){var r=L.createElement("div");return r.innerHTML=c.html,r.children}}}),c}function r6(c){var a=c.children,e=c.main,r=c.mask,s=c.attributes,i=c.styles,f=c.transform;if(z1(f)&&e.found&&!r.found){var n=e.width,l=e.height,o={x:n/l/2,y:.5};s.style=R2(H(H({},i),{},{"transform-origin":"".concat(o.x+f.x/16,"em ").concat(o.y+f.y/16,"em")}))}return[{tag:"svg",attributes:s,children:a}]}function s6(c){var a=c.prefix,e=c.iconName,r=c.children,s=c.attributes,i=c.symbol,f=i===!0?"".concat(a,"-").concat(z.cssPrefix,"-").concat(e):i;return[{tag:"svg",attributes:{style:"display: none;"},children:[{tag:"symbol",attributes:H(H({},s),{},{id:f}),children:r}]}]}function M1(c){var a=c.icons,e=a.main,r=a.mask,s=c.prefix,i=c.iconName,f=c.transform,n=c.symbol,l=c.title,o=c.maskId,t=c.titleId,m=c.extra,v=c.watchable,V=v===void 0?!1:v,p=r.found?r:e,g=p.width,u=p.height,N=s==="fak",M=[z.replacementClass,i?"".concat(z.cssPrefix,"-").concat(i):""].filter(function(A){return m.classes.indexOf(A)===-1}).filter(function(A){return A!==""||!!A}).concat(m.classes).join(" "),d={children:[],attributes:H(H({},m.attributes),{},{"data-prefix":s,"data-icon":i,class:M,role:m.attributes.role||"img",xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 ".concat(g," ").concat(u)})},w=N&&!~m.classes.indexOf("fa-fw")?{width:"".concat(g/u*16*.0625,"em")}:{};V&&(d.attributes[K]=""),l&&(d.children.push({tag:"title",attributes:{id:d.attributes["aria-labelledby"]||"title-".concat(t||h2())},children:[l]}),delete d.attributes.title);var b=H(H({},d),{},{prefix:s,iconName:i,main:e,mask:r,maskId:o,transform:f,symbol:n,styles:H(H({},w),m.styles)}),T=r.found&&e.found?U("generateAbstractMask",b)||{children:[],attributes:{}}:U("generateAbstractIcon",b)||{children:[],attributes:{}},B=T.children,n2=T.attributes;return b.children=B,b.attributes=n2,n?s6(b):r6(b)}function q1(c){var a=c.content,e=c.width,r=c.height,s=c.transform,i=c.title,f=c.extra,n=c.watchable,l=n===void 0?!1:n,o=H(H(H({},f.attributes),i?{title:i}:{}),{},{class:f.classes.join(" ")});l&&(o[K]="");var t=H({},f.styles);z1(s)&&(t.transform=F4({transform:s,startCentered:!0,width:e,height:r}),t["-webkit-transform"]=t.transform);var m=R2(t);m.length>0&&(o.style=m);var v=[];return v.push({tag:"span",attributes:o,children:[a]}),i&&v.push({tag:"span",attributes:{class:"sr-only"},children:[i]}),v}function i6(c){var a=c.content,e=c.title,r=c.extra,s=H(H(H({},r.attributes),e?{title:e}:{}),{},{class:r.classes.join(" ")}),i=R2(r.styles);i.length>0&&(s.style=i);var f=[];return f.push({tag:"span",attributes:s,children:[a]}),e&&f.push({tag:"span",attributes:{class:"sr-only"},children:[e]}),f}var X2=P.styles;function r1(c){var a=c[0],e=c[1],r=c.slice(4),s=n1(r,1),i=s[0],f=null;return Array.isArray(i)?f={tag:"g",attributes:{class:"".concat(z.cssPrefix,"-").concat(X.GROUP)},children:[{tag:"path",attributes:{class:"".concat(z.cssPrefix,"-").concat(X.SECONDARY),fill:"currentColor",d:i[0]}},{tag:"path",attributes:{class:"".concat(z.cssPrefix,"-").concat(X.PRIMARY),fill:"currentColor",d:i[1]}}]}:f={tag:"path",attributes:{fill:"currentColor",d:i}},{found:!0,width:a,height:e,icon:f}}var f6={found:!1,width:512,height:512};function n6(c,a){!l3&&!z.showMissingIcons&&c&&console.error('Icon with name "'.concat(c,'" and prefix "').concat(a,'" is missing.'))}function s1(c,a){var e=a;return a==="fa"&&z.styleDefault!==null&&(a=W()),new Promise(function(r,s){var i={found:!1,width:512,height:512,icon:U("missingIconAbstract")||{}};if(e==="fa"){var f=L3(c)||{};c=f.iconName||c,a=f.prefix||a}if(c&&a&&X2[a]&&X2[a][c]){var n=X2[a][c];return r(r1(n))}n6(c,a),r(H(H({},f6),{},{icon:z.showMissingIcons&&c?U("missingIconAbstract")||{}:{}}))})}var W1=function(){},i1=z.measurePerformance&&d2&&d2.mark&&d2.measure?d2:{mark:W1,measure:W1},o2='FA "6.4.0"',l6=function(a){return i1.mark("".concat(o2," ").concat(a," begins")),function(){return g3(a)}},g3=function(a){i1.mark("".concat(o2," ").concat(a," ends")),i1.measure("".concat(o2," ").concat(a),"".concat(o2," ").concat(a," begins"),"".concat(o2," ").concat(a," ends"))},p1={begin:l6,end:g3},A2=function(){};function G1(c){var a=c.getAttribute?c.getAttribute(K):null;return typeof a=="string"}function o6(c){var a=c.getAttribute?c.getAttribute(o1):null,e=c.getAttribute?c.getAttribute(t1):null;return a&&e}function t6(c){return c&&c.classList&&c.classList.contains&&c.classList.contains(z.replacementClass)}function m6(){if(z.autoReplaceSvg===!0)return P2.replace;var c=P2[z.autoReplaceSvg];return c||P2.replace}function H6(c){return L.createElementNS("http://www.w3.org/2000/svg",c)}function z6(c){return L.createElement(c)}function x3(c){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},e=a.ceFn,r=e===void 0?c.tag==="svg"?H6:z6:e;if(typeof c=="string")return L.createTextNode(c);var s=r(c.tag);Object.keys(c.attributes||[]).forEach(function(f){s.setAttribute(f,c.attributes[f])});var i=c.children||[];return i.forEach(function(f){s.appendChild(x3(f,{ceFn:r}))}),s}function v6(c){var a=" ".concat(c.outerHTML," ");return a="".concat(a,"Font Awesome fontawesome.com "),a}var P2={replace:function(a){var e=a[0];if(e.parentNode)if(a[1].forEach(function(s){e.parentNode.insertBefore(x3(s),e)}),e.getAttribute(K)===null&&z.keepOriginalSource){var r=L.createComment(v6(e));e.parentNode.replaceChild(r,e)}else e.remove()},nest:function(a){var e=a[0],r=a[1];if(~H1(e).indexOf(z.replacementClass))return P2.replace(a);var s=new RegExp("".concat(z.cssPrefix,"-.*"));if(delete r[0].attributes.id,r[0].attributes.class){var i=r[0].attributes.class.split(" ").reduce(function(n,l){return l===z.replacementClass||l.match(s)?n.toSvg.push(l):n.toNode.push(l),n},{toNode:[],toSvg:[]});r[0].attributes.class=i.toSvg.join(" "),i.toNode.length===0?e.removeAttribute("class"):e.setAttribute("class",i.toNode.join(" "))}var f=r.map(function(n){return u2(n)}).join(`
`);e.setAttribute(K,""),e.innerHTML=f}};function _1(c){c()}function N3(c,a){var e=typeof a=="function"?a:A2;if(c.length===0)e();else{var r=_1;z.mutateApproach===u4&&(r=q.requestAnimationFrame||_1),r(function(){var s=m6(),i=p1.begin("mutate");c.map(s),i(),e()})}}var u1=!1;function b3(){u1=!0}function f1(){u1=!1}var F2=null;function j1(c){if(B1&&z.observeMutations){var a=c.treeCallback,e=a===void 0?A2:a,r=c.nodeCallback,s=r===void 0?A2:r,i=c.pseudoElementsCallback,f=i===void 0?A2:i,n=c.observeMutationsRoot,l=n===void 0?L:n;F2=new B1(function(o){if(!u1){var t=W();s2(o).forEach(function(m){if(m.type==="childList"&&m.addedNodes.length>0&&!G1(m.addedNodes[0])&&(z.searchPseudoElements&&f(m.target),e(m.target)),m.type==="attributes"&&m.target.parentNode&&z.searchPseudoElements&&f(m.target.parentNode),m.type==="attributes"&&G1(m.target)&&~b4.indexOf(m.attributeName))if(m.attributeName==="class"&&o6(m.target)){var v=E2(H1(m.target)),V=v.prefix,p=v.iconName;m.target.setAttribute(o1,V||t),p&&m.target.setAttribute(t1,p)}else t6(m.target)&&s(m.target)})}}),O&&F2.observe(l,{childList:!0,attributes:!0,characterData:!0,subtree:!0})}}function V6(){F2&&F2.disconnect()}function h6(c){var a=c.getAttribute("style"),e=[];return a&&(e=a.split(";").reduce(function(r,s){var i=s.split(":"),f=i[0],n=i.slice(1);return f&&n.length>0&&(r[f]=n.join(":").trim()),r},{})),e}function M6(c){var a=c.getAttribute("data-prefix"),e=c.getAttribute("data-icon"),r=c.innerText!==void 0?c.innerText.trim():"",s=E2(H1(c));return s.prefix||(s.prefix=W()),a&&e&&(s.prefix=a,s.iconName=e),s.iconName&&s.prefix||(s.prefix&&r.length>0&&(s.iconName=X4(s.prefix,c.innerText)||V1(s.prefix,Z2(c.innerText))),!s.iconName&&z.autoFetchSvg&&c.firstChild&&c.firstChild.nodeType===Node.TEXT_NODE&&(s.iconName=c.firstChild.data)),s}function p6(c){var a=s2(c.attributes).reduce(function(s,i){return s.name!=="class"&&s.name!=="style"&&(s[i.name]=i.value),s},{}),e=c.getAttribute("title"),r=c.getAttribute("data-fa-title-id");return z.autoA11y&&(e?a["aria-labelledby"]="".concat(z.replacementClass,"-title-").concat(r||h2()):(a["aria-hidden"]="true",a.focusable="false")),a}function u6(){return{iconName:null,title:null,titleId:null,prefix:null,transform:F,symbol:!1,mask:{iconName:null,prefix:null,rest:[]},maskId:null,extra:{classes:[],styles:{},attributes:{}}}}function Y1(c){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{styleParser:!0},e=M6(c),r=e.iconName,s=e.prefix,i=e.rest,f=p6(c),n=a1("parseNodeAttributes",{},c),l=a.styleParser?h6(c):[];return H({iconName:r,title:c.getAttribute("title"),titleId:c.getAttribute("data-fa-title-id"),prefix:s,transform:F,mask:{iconName:null,prefix:null,rest:[]},maskId:null,symbol:!1,extra:{classes:i,styles:l,attributes:f}},n)}var C6=P.styles;function S3(c){var a=z.autoReplaceSvg==="nest"?Y1(c,{styleParser:!1}):Y1(c);return~a.extra.classes.indexOf(o3)?U("generateLayersText",c,a):U("generateSvgReplacementMutation",c,a)}var G=new Set;m1.map(function(c){G.add("fa-".concat(c))});Object.keys(H2[C]).map(G.add.bind(G));Object.keys(H2[x]).map(G.add.bind(G));G=M2(G);function X1(c){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:null;if(!O)return Promise.resolve();var e=L.documentElement.classList,r=function(m){return e.add("".concat(F1,"-").concat(m))},s=function(m){return e.remove("".concat(F1,"-").concat(m))},i=z.autoFetchSvg?G:m1.map(function(t){return"fa-".concat(t)}).concat(Object.keys(C6));i.includes("fa")||i.push("fa");var f=[".".concat(o3,":not([").concat(K,"])")].concat(i.map(function(t){return".".concat(t,":not([").concat(K,"])")})).join(", ");if(f.length===0)return Promise.resolve();var n=[];try{n=s2(c.querySelectorAll(f))}catch{}if(n.length>0)r("pending"),s("complete");else return Promise.resolve();var l=p1.begin("onTree"),o=n.reduce(function(t,m){try{var v=S3(m);v&&t.push(v)}catch(V){l3||V.name==="MissingIcon"&&console.error(V)}return t},[]);return new Promise(function(t,m){Promise.all(o).then(function(v){N3(v,function(){r("active"),r("complete"),s("pending"),typeof a=="function"&&a(),l(),t()})}).catch(function(v){l(),m(v)})})}function L6(c){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:null;S3(c).then(function(e){e&&N3([e],a)})}function d6(c){return function(a){var e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},r=(a||{}).icon?a:e1(a||{}),s=e.mask;return s&&(s=(s||{}).icon?s:e1(s||{})),c(r,H(H({},e),{},{mask:s}))}}var g6=function(a){var e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},r=e.transform,s=r===void 0?F:r,i=e.symbol,f=i===void 0?!1:i,n=e.mask,l=n===void 0?null:n,o=e.maskId,t=o===void 0?null:o,m=e.title,v=m===void 0?null:m,V=e.titleId,p=V===void 0?null:V,g=e.classes,u=g===void 0?[]:g,N=e.attributes,M=N===void 0?{}:N,d=e.styles,w=d===void 0?{}:d;if(a){var b=a.prefix,T=a.iconName,B=a.icon;return U2(H({type:"icon"},a),function(){return Q("beforeDOMElementCreation",{iconDefinition:a,params:e}),z.autoA11y&&(v?M["aria-labelledby"]="".concat(z.replacementClass,"-title-").concat(p||h2()):(M["aria-hidden"]="true",M.focusable="false")),M1({icons:{main:r1(B),mask:l?r1(l.icon):{found:!1,width:null,height:null,icon:{}}},prefix:b,iconName:T,transform:H(H({},F),s),symbol:f,title:v,maskId:t,titleId:p,extra:{attributes:M,styles:w,classes:u}})})}},x6={mixout:function(){return{icon:d6(g6)}},hooks:function(){return{mutationObserverCallbacks:function(e){return e.treeCallback=X1,e.nodeCallback=L6,e}}},provides:function(a){a.i2svg=function(e){var r=e.node,s=r===void 0?L:r,i=e.callback,f=i===void 0?function(){}:i;return X1(s,f)},a.generateSvgReplacementMutation=function(e,r){var s=r.iconName,i=r.title,f=r.titleId,n=r.prefix,l=r.transform,o=r.symbol,t=r.mask,m=r.maskId,v=r.extra;return new Promise(function(V,p){Promise.all([s1(s,n),t.iconName?s1(t.iconName,t.prefix):Promise.resolve({found:!1,width:512,height:512,icon:{}})]).then(function(g){var u=n1(g,2),N=u[0],M=u[1];V([e,M1({icons:{main:N,mask:M},prefix:n,iconName:s,transform:l,symbol:o,maskId:m,title:i,titleId:f,extra:v,watchable:!0})])}).catch(p)})},a.generateAbstractIcon=function(e){var r=e.children,s=e.attributes,i=e.main,f=e.transform,n=e.styles,l=R2(n);l.length>0&&(s.style=l);var o;return z1(f)&&(o=U("generateAbstractTransformGrouping",{main:i,transform:f,containerWidth:i.width,iconWidth:i.width})),r.push(o||i.icon),{children:r,attributes:s}}}},N6={mixout:function(){return{layer:function(e){var r=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},s=r.classes,i=s===void 0?[]:s;return U2({type:"layer"},function(){Q("beforeDOMElementCreation",{assembler:e,params:r});var f=[];return e(function(n){Array.isArray(n)?n.map(function(l){f=f.concat(l.abstract)}):f=f.concat(n.abstract)}),[{tag:"span",attributes:{class:["".concat(z.cssPrefix,"-layers")].concat(M2(i)).join(" ")},children:f}]})}}}},b6={mixout:function(){return{counter:function(e){var r=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},s=r.title,i=s===void 0?null:s,f=r.classes,n=f===void 0?[]:f,l=r.attributes,o=l===void 0?{}:l,t=r.styles,m=t===void 0?{}:t;return U2({type:"counter",content:e},function(){return Q("beforeDOMElementCreation",{content:e,params:r}),i6({content:e.toString(),title:i,extra:{attributes:o,styles:m,classes:["".concat(z.cssPrefix,"-layers-counter")].concat(M2(n))}})})}}}},S6={mixout:function(){return{text:function(e){var r=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},s=r.transform,i=s===void 0?F:s,f=r.title,n=f===void 0?null:f,l=r.classes,o=l===void 0?[]:l,t=r.attributes,m=t===void 0?{}:t,v=r.styles,V=v===void 0?{}:v;return U2({type:"text",content:e},function(){return Q("beforeDOMElementCreation",{content:e,params:r}),q1({content:e,transform:H(H({},F),i),title:n,extra:{attributes:m,styles:V,classes:["".concat(z.cssPrefix,"-layers-text")].concat(M2(o))}})})}}},provides:function(a){a.generateLayersText=function(e,r){var s=r.title,i=r.transform,f=r.extra,n=null,l=null;if(i3){var o=parseInt(getComputedStyle(e).fontSize,10),t=e.getBoundingClientRect();n=t.width/o,l=t.height/o}return z.autoA11y&&!s&&(f.attributes["aria-hidden"]="true"),Promise.resolve([e,q1({content:e.innerHTML,width:n,height:l,transform:i,title:s,extra:f,watchable:!0})])}}},w6=new RegExp('"',"ug"),$1=[1105920,1112319];function k6(c){var a=c.replace(w6,""),e=q4(a,0),r=e>=$1[0]&&e<=$1[1],s=a.length===2?a[0]===a[1]:!1;return{value:Z2(s?a[0]:a),isSecondary:r||s}}function K1(c,a){var e="".concat(p4).concat(a.replace(":","-"));return new Promise(function(r,s){if(c.getAttribute(e)!==null)return r();var i=s2(c.children),f=i.filter(function(B){return B.getAttribute(J2)===a})[0],n=q.getComputedStyle(c,a),l=n.getPropertyValue("font-family").match(g4),o=n.getPropertyValue("font-weight"),t=n.getPropertyValue("content");if(f&&!l)return c.removeChild(f),r();if(l&&t!=="none"&&t!==""){var m=n.getPropertyValue("content"),v=~["Sharp"].indexOf(l[2])?x:C,V=~["Solid","Regular","Light","Thin","Duotone","Brands","Kit"].indexOf(l[2])?z2[v][l[2].toLowerCase()]:x4[v][o],p=k6(m),g=p.value,u=p.isSecondary,N=l[0].startsWith("FontAwesome"),M=V1(V,g),d=M;if(N){var w=$4(g);w.iconName&&w.prefix&&(M=w.iconName,V=w.prefix)}if(M&&!u&&(!f||f.getAttribute(o1)!==V||f.getAttribute(t1)!==d)){c.setAttribute(e,d),f&&c.removeChild(f);var b=u6(),T=b.extra;T.attributes[J2]=a,s1(M,V).then(function(B){var n2=M1(H(H({},b),{},{icons:{main:B,mask:h1()},prefix:V,iconName:d,extra:T,watchable:!0})),A=L.createElement("svg");a==="::before"?c.insertBefore(A,c.firstChild):c.appendChild(A),A.outerHTML=n2.map(function(l2){return u2(l2)}).join(`
`),c.removeAttribute(e),r()}).catch(s)}else r()}else r()})}function y6(c){return Promise.all([K1(c,"::before"),K1(c,"::after")])}function A6(c){return c.parentNode!==document.head&&!~C4.indexOf(c.tagName.toUpperCase())&&!c.getAttribute(J2)&&(!c.parentNode||c.parentNode.tagName!=="svg")}function Q1(c){if(O)return new Promise(function(a,e){var r=s2(c.querySelectorAll("*")).filter(A6).map(y6),s=p1.begin("searchPseudoElements");b3(),Promise.all(r).then(function(){s(),f1(),a()}).catch(function(){s(),f1(),e()})})}var P6={hooks:function(){return{mutationObserverCallbacks:function(e){return e.pseudoElementsCallback=Q1,e}}},provides:function(a){a.pseudoElements2svg=function(e){var r=e.node,s=r===void 0?L:r;z.searchPseudoElements&&Q1(s)}}},J1=!1,T6={mixout:function(){return{dom:{unwatch:function(){b3(),J1=!0}}}},hooks:function(){return{bootstrap:function(){j1(a1("mutationObserverCallbacks",{}))},noAuto:function(){V6()},watch:function(e){var r=e.observeMutationsRoot;J1?f1():j1(a1("mutationObserverCallbacks",{observeMutationsRoot:r}))}}}},Z1=function(a){var e={size:16,x:0,y:0,flipX:!1,flipY:!1,rotate:0};return a.toLowerCase().split(" ").reduce(function(r,s){var i=s.toLowerCase().split("-"),f=i[0],n=i.slice(1).join("-");if(f&&n==="h")return r.flipX=!0,r;if(f&&n==="v")return r.flipY=!0,r;if(n=parseFloat(n),isNaN(n))return r;switch(f){case"grow":r.size=r.size+n;break;case"shrink":r.size=r.size-n;break;case"left":r.x=r.x-n;break;case"right":r.x=r.x+n;break;case"up":r.y=r.y-n;break;case"down":r.y=r.y+n;break;case"rotate":r.rotate=r.rotate+n;break}return r},e)},B6={mixout:function(){return{parse:{transform:function(e){return Z1(e)}}}},hooks:function(){return{parseNodeAttributes:function(e,r){var s=r.getAttribute("data-fa-transform");return s&&(e.transform=Z1(s)),e}}},provides:function(a){a.generateAbstractTransformGrouping=function(e){var r=e.main,s=e.transform,i=e.containerWidth,f=e.iconWidth,n={transform:"translate(".concat(i/2," 256)")},l="translate(".concat(s.x*32,", ").concat(s.y*32,") "),o="scale(".concat(s.size/16*(s.flipX?-1:1),", ").concat(s.size/16*(s.flipY?-1:1),") "),t="rotate(".concat(s.rotate," 0 0)"),m={transform:"".concat(l," ").concat(o," ").concat(t)},v={transform:"translate(".concat(f/2*-1," -256)")},V={outer:n,inner:m,path:v};return{tag:"g",attributes:H({},V.outer),children:[{tag:"g",attributes:H({},V.inner),children:[{tag:r.icon.tag,children:r.icon.children,attributes:H(H({},r.icon.attributes),V.path)}]}]}}}},$2={x:0,y:0,width:"100%",height:"100%"};function c3(c){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!0;return c.attributes&&(c.attributes.fill||a)&&(c.attributes.fill="black"),c}function F6(c){return c.tag==="g"?c.children:[c]}var R6={hooks:function(){return{parseNodeAttributes:function(e,r){var s=r.getAttribute("data-fa-mask"),i=s?E2(s.split(" ").map(function(f){return f.trim()})):h1();return i.prefix||(i.prefix=W()),e.mask=i,e.maskId=r.getAttribute("data-fa-mask-id"),e}}},provides:function(a){a.generateAbstractMask=function(e){var r=e.children,s=e.attributes,i=e.main,f=e.mask,n=e.maskId,l=e.transform,o=i.width,t=i.icon,m=f.width,v=f.icon,V=B4({transform:l,containerWidth:m,iconWidth:o}),p={tag:"rect",attributes:H(H({},$2),{},{fill:"white"})},g=t.children?{children:t.children.map(c3)}:{},u={tag:"g",attributes:H({},V.inner),children:[c3(H({tag:t.tag,attributes:H(H({},t.attributes),V.path)},g))]},N={tag:"g",attributes:H({},V.outer),children:[u]},M="mask-".concat(n||h2()),d="clip-".concat(n||h2()),w={tag:"mask",attributes:H(H({},$2),{},{id:M,maskUnits:"userSpaceOnUse",maskContentUnits:"userSpaceOnUse"}),children:[p,N]},b={tag:"defs",children:[{tag:"clipPath",attributes:{id:d},children:F6(v)},w]};return r.push(b,{tag:"rect",attributes:H({fill:"currentColor","clip-path":"url(#".concat(d,")"),mask:"url(#".concat(M,")")},$2)}),{children:r,attributes:s}}}},D6={provides:function(a){var e=!1;q.matchMedia&&(e=q.matchMedia("(prefers-reduced-motion: reduce)").matches),a.missingIconAbstract=function(){var r=[],s={fill:"currentColor"},i={attributeType:"XML",repeatCount:"indefinite",dur:"2s"};r.push({tag:"path",attributes:H(H({},s),{},{d:"M156.5,447.7l-12.6,29.5c-18.7-9.5-35.9-21.2-51.5-34.9l22.7-22.7C127.6,430.5,141.5,440,156.5,447.7z M40.6,272H8.5 c1.4,21.2,5.4,41.7,11.7,61.1L50,321.2C45.1,305.5,41.8,289,40.6,272z M40.6,240c1.4-18.8,5.2-37,11.1-54.1l-29.5-12.6 C14.7,194.3,10,216.7,8.5,240H40.6z M64.3,156.5c7.8-14.9,17.2-28.8,28.1-41.5L69.7,92.3c-13.7,15.6-25.5,32.8-34.9,51.5 L64.3,156.5z M397,419.6c-13.9,12-29.4,22.3-46.1,30.4l11.9,29.8c20.7-9.9,39.8-22.6,56.9-37.6L397,419.6z M115,92.4 c13.9-12,29.4-22.3,46.1-30.4l-11.9-29.8c-20.7,9.9-39.8,22.6-56.8,37.6L115,92.4z M447.7,355.5c-7.8,14.9-17.2,28.8-28.1,41.5 l22.7,22.7c13.7-15.6,25.5-32.9,34.9-51.5L447.7,355.5z M471.4,272c-1.4,18.8-5.2,37-11.1,54.1l29.5,12.6 c7.5-21.1,12.2-43.5,13.6-66.8H471.4z M321.2,462c-15.7,5-32.2,8.2-49.2,9.4v32.1c21.2-1.4,41.7-5.4,61.1-11.7L321.2,462z M240,471.4c-18.8-1.4-37-5.2-54.1-11.1l-12.6,29.5c21.1,7.5,43.5,12.2,66.8,13.6V471.4z M462,190.8c5,15.7,8.2,32.2,9.4,49.2h32.1 c-1.4-21.2-5.4-41.7-11.7-61.1L462,190.8z M92.4,397c-12-13.9-22.3-29.4-30.4-46.1l-29.8,11.9c9.9,20.7,22.6,39.8,37.6,56.9 L92.4,397z M272,40.6c18.8,1.4,36.9,5.2,54.1,11.1l12.6-29.5C317.7,14.7,295.3,10,272,8.5V40.6z M190.8,50 c15.7-5,32.2-8.2,49.2-9.4V8.5c-21.2,1.4-41.7,5.4-61.1,11.7L190.8,50z M442.3,92.3L419.6,115c12,13.9,22.3,29.4,30.5,46.1 l29.8-11.9C470,128.5,457.3,109.4,442.3,92.3z M397,92.4l22.7-22.7c-15.6-13.7-32.8-25.5-51.5-34.9l-12.6,29.5 C370.4,72.1,384.4,81.5,397,92.4z"})});var f=H(H({},i),{},{attributeName:"opacity"}),n={tag:"circle",attributes:H(H({},s),{},{cx:"256",cy:"364",r:"28"}),children:[]};return e||n.children.push({tag:"animate",attributes:H(H({},i),{},{attributeName:"r",values:"28;14;28;28;14;28;"})},{tag:"animate",attributes:H(H({},f),{},{values:"1;0;1;1;0;1;"})}),r.push(n),r.push({tag:"path",attributes:H(H({},s),{},{opacity:"1",d:"M263.7,312h-16c-6.6,0-12-5.4-12-12c0-71,77.4-63.9,77.4-107.8c0-20-17.8-40.2-57.4-40.2c-29.1,0-44.3,9.6-59.2,28.7 c-3.9,5-11.1,6-16.2,2.4l-13.1-9.2c-5.6-3.9-6.9-11.8-2.6-17.2c21.2-27.2,46.4-44.7,91.2-44.7c52.3,0,97.4,29.8,97.4,80.2 c0,67.6-77.4,63.5-77.4,107.8C275.7,306.6,270.3,312,263.7,312z"}),children:e?[]:[{tag:"animate",attributes:H(H({},f),{},{values:"1;0;0;0;0;1;"})}]}),e||r.push({tag:"path",attributes:H(H({},s),{},{opacity:"0",d:"M232.5,134.5l7,168c0.3,6.4,5.6,11.5,12,11.5h9c6.4,0,11.7-5.1,12-11.5l7-168c0.3-6.8-5.2-12.5-12-12.5h-23 C237.7,122,232.2,127.7,232.5,134.5z"}),children:[{tag:"animate",attributes:H(H({},f),{},{values:"0;0;1;1;0;0;"})}]}),{tag:"g",attributes:{class:"missing"},children:r}}}},E6={hooks:function(){return{parseNodeAttributes:function(e,r){var s=r.getAttribute("data-fa-symbol"),i=s===null?!1:s===""?!0:s;return e.symbol=i,e}}}},U6=[D4,x6,N6,b6,S6,P6,T6,B6,R6,D6,E6];J4(U6,{mixoutsTo:y});var u0=y.noAuto,C0=y.config,L0=y.library,d0=y.dom,O2=y.parse,g0=y.findIconDefinition,x0=y.toHtml,w3=y.icon,N0=y.layer,b0=y.text,S0=y.counter;var h=J3(R3());import U3 from"react";function D3(c,a){var e=Object.keys(c);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(c);a&&(r=r.filter(function(s){return Object.getOwnPropertyDescriptor(c,s).enumerable})),e.push.apply(e,r)}return e}function _(c){for(var a=1;a<arguments.length;a++){var e=arguments[a]!=null?arguments[a]:{};a%2?D3(Object(e),!0).forEach(function(r){i2(c,r,e[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(c,Object.getOwnPropertyDescriptors(e)):D3(Object(e)).forEach(function(r){Object.defineProperty(c,r,Object.getOwnPropertyDescriptor(e,r))})}return c}function I2(c){"@babel/helpers - typeof";return I2=typeof Symbol=="function"&&typeof Symbol.iterator=="symbol"?function(a){return typeof a}:function(a){return a&&typeof Symbol=="function"&&a.constructor===Symbol&&a!==Symbol.prototype?"symbol":typeof a},I2(c)}function i2(c,a,e){return a in c?Object.defineProperty(c,a,{value:e,enumerable:!0,configurable:!0,writable:!0}):c[a]=e,c}function q6(c,a){if(c==null)return{};var e={},r=Object.keys(c),s,i;for(i=0;i<r.length;i++)s=r[i],!(a.indexOf(s)>=0)&&(e[s]=c[s]);return e}function W6(c,a){if(c==null)return{};var e=q6(c,a),r,s;if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(c);for(s=0;s<i.length;s++)r=i[s],!(a.indexOf(r)>=0)&&Object.prototype.propertyIsEnumerable.call(c,r)&&(e[r]=c[r])}return e}function L1(c){return G6(c)||_6(c)||j6(c)||Y6()}function G6(c){if(Array.isArray(c))return d1(c)}function _6(c){if(typeof Symbol<"u"&&c[Symbol.iterator]!=null||c["@@iterator"]!=null)return Array.from(c)}function j6(c,a){if(c){if(typeof c=="string")return d1(c,a);var e=Object.prototype.toString.call(c).slice(8,-1);if(e==="Object"&&c.constructor&&(e=c.constructor.name),e==="Map"||e==="Set")return Array.from(c);if(e==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(e))return d1(c,a)}}function d1(c,a){(a==null||a>c.length)&&(a=c.length);for(var e=0,r=new Array(a);e<a;e++)r[e]=c[e];return r}function Y6(){throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function X6(c){var a,e=c.beat,r=c.fade,s=c.beatFade,i=c.bounce,f=c.shake,n=c.flash,l=c.spin,o=c.spinPulse,t=c.spinReverse,m=c.pulse,v=c.fixedWidth,V=c.inverse,p=c.border,g=c.listItem,u=c.flip,N=c.size,M=c.rotation,d=c.pull,w=(a={"fa-beat":e,"fa-fade":r,"fa-beat-fade":s,"fa-bounce":i,"fa-shake":f,"fa-flash":n,"fa-spin":l,"fa-spin-reverse":t,"fa-spin-pulse":o,"fa-pulse":m,"fa-fw":v,"fa-inverse":V,"fa-border":p,"fa-li":g,"fa-flip":u===!0,"fa-flip-horizontal":u==="horizontal"||u==="both","fa-flip-vertical":u==="vertical"||u==="both"},i2(a,"fa-".concat(N),typeof N<"u"&&N!==null),i2(a,"fa-rotate-".concat(M),typeof M<"u"&&M!==null&&M!==0),i2(a,"fa-pull-".concat(d),typeof d<"u"&&d!==null),i2(a,"fa-swap-opacity",c.swapOpacity),a);return Object.keys(w).map(function(b){return w[b]?b:null}).filter(function(b){return b})}function $6(c){return c=c-0,c===c}function O3(c){return $6(c)?c:(c=c.replace(/[\-_\s]+(.)?/g,function(a,e){return e?e.toUpperCase():""}),c.substr(0,1).toLowerCase()+c.substr(1))}var K6=["style"];function Q6(c){return c.charAt(0).toUpperCase()+c.slice(1)}function J6(c){return c.split(";").map(function(a){return a.trim()}).filter(function(a){return a}).reduce(function(a,e){var r=e.indexOf(":"),s=O3(e.slice(0,r)),i=e.slice(r+1).trim();return s.startsWith("webkit")?a[Q6(s)]=i:a[s]=i,a},{})}function I3(c,a){var e=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{};if(typeof a=="string")return a;var r=(a.children||[]).map(function(l){return I3(c,l)}),s=Object.keys(a.attributes||{}).reduce(function(l,o){var t=a.attributes[o];switch(o){case"class":l.attrs.className=t,delete a.attributes.class;break;case"style":l.attrs.style=J6(t);break;default:o.indexOf("aria-")===0||o.indexOf("data-")===0?l.attrs[o.toLowerCase()]=t:l.attrs[O3(o)]=t}return l},{attrs:{}}),i=e.style,f=i===void 0?{}:i,n=W6(e,K6);return s.attrs.style=_(_({},s.attrs.style),f),c.apply(void 0,[a.tag,_(_({},s.attrs),n)].concat(L1(r)))}var q3=!1;try{q3=!0}catch{}function Z6(){if(!q3&&console&&typeof console.error=="function"){var c;(c=console).error.apply(c,arguments)}}function E3(c){if(c&&I2(c)==="object"&&c.prefix&&c.iconName&&c.icon)return c;if(O2.icon)return O2.icon(c);if(c===null)return null;if(c&&I2(c)==="object"&&c.prefix&&c.iconName)return c;if(Array.isArray(c)&&c.length===2)return{prefix:c[0],iconName:c[1]};if(typeof c=="string")return{prefix:"fas",iconName:c}}function C1(c,a){return Array.isArray(a)&&a.length>0||!Array.isArray(a)&&a?i2({},c,a):{}}var j=U3.forwardRef(function(c,a){var e=c.icon,r=c.mask,s=c.symbol,i=c.className,f=c.title,n=c.titleId,l=c.maskId,o=E3(e),t=C1("classes",[].concat(L1(X6(c)),L1(i.split(" ")))),m=C1("transform",typeof c.transform=="string"?O2.transform(c.transform):c.transform),v=C1("mask",E3(r)),V=w3(o,_(_(_(_({},t),m),v),{},{symbol:s,title:f,titleId:n,maskId:l}));if(!V)return Z6("Could not find icon",o),null;var p=V.abstract,g={ref:a};return Object.keys(c).forEach(function(u){j.defaultProps.hasOwnProperty(u)||(g[u]=c[u])}),c0(p[0],g)});j.displayName="FontAwesomeIcon";j.propTypes={beat:h.default.bool,border:h.default.bool,beatFade:h.default.bool,bounce:h.default.bool,className:h.default.string,fade:h.default.bool,flash:h.default.bool,mask:h.default.oneOfType([h.default.object,h.default.array,h.default.string]),maskId:h.default.string,fixedWidth:h.default.bool,inverse:h.default.bool,flip:h.default.oneOf([!0,!1,"horizontal","vertical","both"]),icon:h.default.oneOfType([h.default.object,h.default.array,h.default.string]),listItem:h.default.bool,pull:h.default.oneOf(["right","left"]),pulse:h.default.bool,rotation:h.default.oneOf([0,90,180,270]),shake:h.default.bool,size:h.default.oneOf(["2xs","xs","sm","lg","xl","2xl","1x","2x","3x","4x","5x","6x","7x","8x","9x","10x"]),spin:h.default.bool,spinPulse:h.default.bool,spinReverse:h.default.bool,symbol:h.default.oneOfType([h.default.bool,h.default.string]),title:h.default.string,titleId:h.default.string,transform:h.default.oneOfType([h.default.string,h.default.object]),swapOpacity:h.default.bool};j.defaultProps={border:!1,className:"",mask:null,maskId:null,fixedWidth:!1,inverse:!1,flip:!1,icon:null,listItem:!1,pull:null,pulse:!1,rotation:null,size:null,spin:!1,spinPulse:!1,spinReverse:!1,beat:!1,fade:!1,beatFade:!1,bounce:!1,shake:!1,symbol:!1,title:"",titleId:null,transform:null,swapOpacity:!1};var c0=I3.bind(null,U3.createElement);var a0={prefix:"fas",iconName:"face-meh",icon:[512,512,[128528,"meh"],"f11a","M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM176.4 176a32 32 0 1 1 0 64 32 32 0 1 1 0-64zm128 32a32 32 0 1 1 64 0 32 32 0 1 1 -64 0zM160 336H352c8.8 0 16 7.2 16 16s-7.2 16-16 16H160c-8.8 0-16-7.2-16-16s7.2-16 16-16z"]},W3=a0;var G3={prefix:"fas",iconName:"thumbs-down",icon:[512,512,[128078,61576],"f165","M313.4 479.1c26-5.2 42.9-30.5 37.7-56.5l-2.3-11.4c-5.3-26.7-15.1-52.1-28.8-75.2H464c26.5 0 48-21.5 48-48c0-18.5-10.5-34.6-25.9-42.6C497 236.6 504 223.1 504 208c0-23.4-16.8-42.9-38.9-47.1c4.4-7.3 6.9-15.8 6.9-24.9c0-21.3-13.9-39.4-33.1-45.6c.7-3.3 1.1-6.8 1.1-10.4c0-26.5-21.5-48-48-48H294.5c-19 0-37.5 5.6-53.3 16.1L202.7 73.8C176 91.6 160 121.6 160 153.7V192v48 24.9c0 29.2 13.3 56.7 36 75l7.4 5.9c26.5 21.2 44.6 51 51.2 84.2l2.3 11.4c5.2 26 30.5 42.9 56.5 37.7zM32 384H96c17.7 0 32-14.3 32-32V128c0-17.7-14.3-32-32-32H32C14.3 96 0 110.3 0 128V352c0 17.7 14.3 32 32 32z"]};var _3={prefix:"fas",iconName:"thumbs-up",icon:[512,512,[128077,61575],"f164","M313.4 32.9c26 5.2 42.9 30.5 37.7 56.5l-2.3 11.4c-5.3 26.7-15.1 52.1-28.8 75.2H464c26.5 0 48 21.5 48 48c0 18.5-10.5 34.6-25.9 42.6C497 275.4 504 288.9 504 304c0 23.4-16.8 42.9-38.9 47.1c4.4 7.3 6.9 15.8 6.9 24.9c0 21.3-13.9 39.4-33.1 45.6c.7 3.3 1.1 6.8 1.1 10.4c0 26.5-21.5 48-48 48H294.5c-19 0-37.5-5.6-53.3-16.1l-38.5-25.7C176 420.4 160 390.4 160 358.3V320 272 247.1c0-29.2 13.3-56.7 36-75l7.4-5.9c26.5-21.2 44.6-51 51.2-84.2l2.3-11.4c5.2-26 30.5-42.9 56.5-37.7zM32 192H96c17.7 0 32 14.3 32 32V448c0 17.7-14.3 32-32 32H32c-17.7 0-32-14.3-32-32V224c0-17.7 14.3-32 32-32z"]};import{jsx as k,jsxs as C2}from"react/jsx-runtime";var W2=new f0({box:{low:{dtype:r0.DTYPE_FLOAT32,shape:[1]}}}),X3=({sendAction:c,fps:a=30,actorClass:e,observation:r,tickId:s,...i})=>{let[f,n]=f2(!1),[l,o]=f2(!0),[t,m]=f2(2),[v,V]=f2(!1),[p,g]=f2(""),[u,N]=f2(!1),M=r?.gamePlayerName,d=r?.currentPlayer?.name,w=r?.actionValue,b=["NONE","FIRE","UP","DOWN","FIRE UP"," FIRE DOWN"];g1(()=>{M&&(M.includes("first")?g("right"):g("left"))},[M]),g1(()=>{d==o0?V(!0):V(!1)},[d]),g1(()=>{!l||!v||s%3!=0?N(!1):N(!0)},[s,l,v]);let T=j3(()=>n(R=>!R),[n]),B=j3(()=>o(R=>!R),[o]);Y3("p",T),Y3("h",B);let{currentFps:n2}=l0(R=>{},a,f);if(!l||!v||s%3!=0){let R=q2(W2,[0]);c(R)}let A=R=>{let L2;R=="LIKE"?(m(1),L2=q2(W2,[1])):R=="DISLIKE"?(m(-1),L2=q2(W2,[-1])):(m(0),L2=q2(W2,[0])),setTimeout(()=>{m(2)},150),c(L2)},l2="opacity-90 py-1 rounded-full items-center text-white px-2  text-sm outline-none",K3=`bg-green-500 ${l2}`,Q3=`bg-orange-500 ${l2}`;return C2("div",{...i,children:[u?C2("div",{className:"flex p-2 flex-row justify-center items-center",children:[k("div",{className:p=="right"?K3:Q3,children:M}),k("div",{className:`bg-purple-500 ${l2}`,children:b[w]})]}):k("div",{}),C2("div",{className:"flex p-2 flex-row justify-center items-center",children:[k("div",{className:"flex justify-center p-2",children:k(n0,{check:l,onChange:B,label:"Human feedback"})}),l&&C2("div",{className:"ml-5 text-center",style:{paddingBottom:10,padingTop:10},children:[k("button",{type:"button",onClick:()=>A("LIKE"),children:k(j,{icon:_3,style:{paddingRight:5},size:"2x",color:t==1?"green":"gray"})}),k("button",{type:"button",onClick:()=>A("NEURAL"),children:k(j,{icon:W3,style:{paddingLeft:5},size:"2x",color:t==0?"blue":"gray"})}),k("button",{type:"button",onClick:()=>A("DISLIKE"),children:k(j,{icon:G3,style:{paddingLeft:15},size:"2x",color:t==-1?"red":"gray"})})]})]}),C2("div",{className:"flex flex-row gap-1 p-3",children:[k(e0,{className:"flex-1",onClick:T,children:f?"Resume":"Pause"}),k(s0,{className:"flex-none w-fit",value:n2})]}),k(i0,{items:[["p","Pause/Unpause"],["h","Feedback/Not Feedback"]]})]})};import{jsx as x1}from"react/jsx-runtime";var z0=({actorParams:c,...a})=>{let e=c?.className;return e===t0?x1(H0,{actorParams:c,...a}):e===m0?x1($3,{actorParams:c,...a,controls:X3}):x1($3,{actorParams:c,...a,controls:w1})},Y0=z0;export{Y0 as default};
//# sourceMappingURL=AtariPong.js.map
