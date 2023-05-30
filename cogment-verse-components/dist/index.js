import"react";import*as v from"react";import*as h from"react";function fe(){return fe=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var r=arguments[t];for(var a in r)Object.prototype.hasOwnProperty.call(r,a)&&(e[a]=r[a])}return e},fe.apply(this,arguments)}var z;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(z||(z={}));function x(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function M(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function $(e){let{pathname:t="/",search:r="",hash:a=""}=e;return r&&r!=="?"&&(t+=r.charAt(0)==="?"?r:"?"+r),a&&a!=="#"&&(t+=a.charAt(0)==="#"?a:"#"+a),t}function W(e){let t={};if(e){let r=e.indexOf("#");r>=0&&(t.hash=e.substr(r),e=e.substr(0,r));let a=e.indexOf("?");a>=0&&(t.search=e.substr(a),e=e.substr(0,a)),e&&(t.pathname=e)}return t}var Ce;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(Ce||(Ce={}));function I(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let r=t.endsWith("/")?t.length-1:t.length,a=e.charAt(r);return a&&a!=="/"?null:e.slice(r)||"/"}function Pe(e,t){t===void 0&&(t="/");let{pathname:r,search:a="",hash:n=""}=typeof e=="string"?W(e):e;return{pathname:r?r.startsWith("/")?r:dt(r,t):t,search:ft(a),hash:ht(n)}}function dt(e,t){let r=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(n=>{n===".."?r.length>1&&r.pop():n!=="."&&r.push(n)}),r.length>1?r.join("/"):"/"}function de(e,t,r,a){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(a)+"].  Please separate it out to the ")+("`to."+r+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function he(e){return e.filter((t,r)=>r===0||t.route.path&&t.route.path.length>0)}function pe(e,t,r,a){a===void 0&&(a=!1);let n;typeof e=="string"?n=W(e):(n=fe({},e),x(!n.pathname||!n.pathname.includes("?"),de("?","pathname","search",n)),x(!n.pathname||!n.pathname.includes("#"),de("#","pathname","hash",n)),x(!n.search||!n.search.includes("#"),de("#","search","hash",n)));let i=e===""||n.pathname==="",o=i?"/":n.pathname,l;if(a||o==null)l=r;else{let d=t.length-1;if(o.startsWith("..")){let m=o.split("/");for(;m[0]==="..";)m.shift(),d-=1;n.pathname=m.join("/")}l=d>=0?t[d]:"/"}let s=Pe(n,l),u=o&&o!=="/"&&o.endsWith("/"),p=(i||o===".")&&r.endsWith("/");return!s.pathname.endsWith("/")&&(u||p)&&(s.pathname+="/"),s}var K=e=>e.join("/").replace(/\/\/+/g,"/");var ft=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,ht=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;var Ae=["post","put","patch","delete"],$r=new Set(Ae),pt=["get",...Ae],Wr=new Set(pt);var Kr=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u";var Jr=Symbol("deferred");function me(){return me=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var r=arguments[t];for(var a in r)Object.prototype.hasOwnProperty.call(r,a)&&(e[a]=r[a])}return e},me.apply(this,arguments)}var J=h.createContext(null);J.displayName="DataRouter";var Y=h.createContext(null);Y.displayName="DataRouterState";var Rt=h.createContext(null);Rt.displayName="Await";var C=h.createContext(null);C.displayName="Navigation";var Z=h.createContext(null);Z.displayName="Location";var F=h.createContext({outlet:null,matches:[],isDataRoute:!1});F.displayName="Route";var _t=h.createContext(null);_t.displayName="RouteError";function Ue(e,t){let{relative:r}=t===void 0?{}:t;ee()||x(!1,"useHref() may be used only in the context of a <Router> component.");let{basename:a,navigator:n}=h.useContext(C),{hash:i,pathname:o,search:l}=G(e,{relative:r}),s=o;return a!=="/"&&(s=o==="/"?a:K([a,o])),n.createHref({pathname:s,search:l,hash:i})}function ee(){return h.useContext(Z)!=null}function O(){return ee()||x(!1,"useLocation() may be used only in the context of a <Router> component."),h.useContext(Z).location}var ke="You should call navigate() in a React.useEffect(), not when your component is first rendered.";function Fe(e){h.useContext(C).static||h.useLayoutEffect(e)}function Oe(){let{isDataRoute:e}=h.useContext(F);return e?Nt():Et()}function Et(){ee()||x(!1,"useNavigate() may be used only in the context of a <Router> component.");let e=h.useContext(J),{basename:t,navigator:r}=h.useContext(C),{matches:a}=h.useContext(F),{pathname:n}=O(),i=JSON.stringify(he(a).map(s=>s.pathnameBase)),o=h.useRef(!1);return Fe(()=>{o.current=!0}),h.useCallback(function(s,u){if(u===void 0&&(u={}),M(o.current,ke),!o.current)return;if(typeof s=="number"){r.go(s);return}let p=pe(s,JSON.parse(i),n,u.relative==="path");e==null&&t!=="/"&&(p.pathname=p.pathname==="/"?t:K([t,p.pathname])),(u.replace?r.replace:r.push)(p,u.state,u)},[t,r,i,n,e])}function G(e,t){let{relative:r}=t===void 0?{}:t,{matches:a}=h.useContext(F),{pathname:n}=O(),i=JSON.stringify(he(a).map(o=>o.pathnameBase));return h.useMemo(()=>pe(e,JSON.parse(i),n,r==="path"),[e,i,n,r])}var ve;(function(e){e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate"})(ve||(ve={}));var j;(function(e){e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId"})(j||(j={}));function ge(e){return e+" must be used within a data router.  See https://reactrouter.com/routers/picking-a-router."}function wt(e){let t=h.useContext(J);return t||x(!1,ge(e)),t}function Te(e){let t=h.useContext(Y);return t||x(!1,ge(e)),t}function xt(e){let t=h.useContext(F);return t||x(!1,ge(e)),t}function Me(e){let t=xt(e),r=t.matches[t.matches.length-1];return r.route.id||x(!1,e+' can only be used on routes that contain a unique "id"'),r.route.id}function Ie(){return Me(j.UseRouteId)}function je(){return Te(j.UseNavigation).navigation}function Be(){let{matches:e,loaderData:t}=Te(j.UseMatches);return h.useMemo(()=>e.map(r=>{let{pathname:a,params:n}=r;return{id:r.route.id,pathname:a,params:n,data:t[r.route.id],handle:r.route.handle}}),[e,t])}function Nt(){let{router:e}=wt(ve.UseNavigateStable),t=Me(j.UseNavigateStable),r=h.useRef(!1);return Fe(()=>{r.current=!0}),h.useCallback(function(n,i){i===void 0&&(i={}),M(r.current,ke),r.current&&(typeof n=="number"?e.navigate(n):e.navigate(n,me({fromRouteId:t},i)))},[e,t])}function Ve(e){let{basename:t="/",children:r=null,location:a,navigationType:n=z.Pop,navigator:i,static:o=!1}=e;ee()&&x(!1,"You cannot render a <Router> inside another <Router>. You should never have more than one in your app.");let l=t.replace(/^\/*/,"/"),s=h.useMemo(()=>({basename:l,navigator:i,static:o}),[l,i,o]);typeof a=="string"&&(a=W(a));let{pathname:u="/",search:p="",hash:d="",state:m=null,key:g="default"}=a,_=h.useMemo(()=>{let b=I(u,l);return b==null?null:{location:{pathname:b,search:p,hash:d,state:m,key:g},navigationType:n}},[l,u,p,d,m,g,n]);return M(_!=null,'<Router basename="'+l+'"> is not able to match the URL '+('"'+u+p+d+'" because it does not start with the ')+"basename, so the <Router> won't render anything."),_==null?null:h.createElement(C.Provider,{value:s},h.createElement(Z.Provider,{children:r,value:_}))}var Le;(function(e){e[e.pending=0]="pending",e[e.success=1]="success",e[e.error=2]="error"})(Le||(Le={}));var ta=new Promise(()=>{});function A(){return A=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var r=arguments[t];for(var a in r)Object.prototype.hasOwnProperty.call(r,a)&&(e[a]=r[a])}return e},A.apply(this,arguments)}function Re(e,t){if(e==null)return{};var r={},a=Object.keys(e),n,i;for(i=0;i<a.length;i++)n=a[i],!(t.indexOf(n)>=0)&&(r[n]=e[n]);return r}var re="get",ye="application/x-www-form-urlencoded";function ne(e){return e!=null&&typeof e.tagName=="string"}function Dt(e){return ne(e)&&e.tagName.toLowerCase()==="button"}function St(e){return ne(e)&&e.tagName.toLowerCase()==="form"}function Ct(e){return ne(e)&&e.tagName.toLowerCase()==="input"}function Pt(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function At(e,t){return e.button===0&&(!t||t==="_self")&&!Pt(e)}function Lt(e,t,r){let a,n=null,i,o;if(St(e)){let l=t.submissionTrigger;if(t.action)n=t.action;else{let s=e.getAttribute("action");n=s?I(s,r):null}a=t.method||e.getAttribute("method")||re,i=t.encType||e.getAttribute("enctype")||ye,o=new FormData(e),l&&l.name&&o.append(l.name,l.value)}else if(Dt(e)||Ct(e)&&(e.type==="submit"||e.type==="image")){let l=e.form;if(l==null)throw new Error('Cannot submit a <button> or <input type="submit"> without a <form>');if(t.action)n=t.action;else{let s=e.getAttribute("formaction")||l.getAttribute("action");n=s?I(s,r):null}a=t.method||e.getAttribute("formmethod")||l.getAttribute("method")||re,i=t.encType||e.getAttribute("formenctype")||l.getAttribute("enctype")||ye,o=new FormData(l),e.name&&o.append(e.name,e.value)}else{if(ne(e))throw new Error('Cannot submit element that is not <form>, <button>, or <input type="submit|image">');if(a=t.method||re,n=t.action||null,i=t.encType||ye,e instanceof FormData)o=e;else if(o=new FormData,e instanceof URLSearchParams)for(let[l,s]of e)o.append(l,s);else if(e!=null)for(let l of Object.keys(e))o.append(l,e[l])}return{action:n,method:a.toLowerCase(),encType:i,formData:o}}var Ut=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset"],kt=["aria-current","caseSensitive","className","end","style","to","children"],Ft=["reloadDocument","replace","method","action","onSubmit","fetcherKey","routeId","relative","preventScrollReset"];function Ot(e){let{basename:t,children:r,history:a}=e,[n,i]=v.useState({action:a.action,location:a.location});return v.useLayoutEffect(()=>a.listen(i),[a]),v.createElement(Ve,{basename:t,children:r,location:n.location,navigationType:n.action,navigator:a})}Ot.displayName="unstable_HistoryRouter";var Tt=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",Mt=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,B=v.forwardRef(function(t,r){let{onClick:a,relative:n,reloadDocument:i,replace:o,state:l,target:s,to:u,preventScrollReset:p}=t,d=Re(t,Ut),{basename:m}=v.useContext(C),g,_=!1;if(typeof u=="string"&&Mt.test(u)&&(g=u,Tt))try{let c=new URL(window.location.href),y=u.startsWith("//")?new URL(c.protocol+u):new URL(u),N=I(y.pathname,m);y.origin===c.origin&&N!=null?u=N+y.search+y.hash:_=!0}catch{M(!1,'<Link to="'+u+'"> contains an invalid URL which will probably break when clicked - please update to a valid URL path.')}let b=Ue(u,{relative:n}),R=Ht(u,{replace:o,state:l,target:s,preventScrollReset:p,relative:n});function f(c){a&&a(c),c.defaultPrevented||R(c)}return v.createElement("a",A({},d,{href:g||b,onClick:_||i?a:f,ref:r,target:s}))});B.displayName="Link";var It=v.forwardRef(function(t,r){let{"aria-current":a="page",caseSensitive:n=!1,className:i="",end:o=!1,style:l,to:s,children:u}=t,p=Re(t,kt),d=G(s,{relative:p.relative}),m=O(),g=v.useContext(Y),{navigator:_}=v.useContext(C),b=_.encodeLocation?_.encodeLocation(d).pathname:d.pathname,R=m.pathname,f=g&&g.navigation&&g.navigation.location?g.navigation.location.pathname:null;n||(R=R.toLowerCase(),f=f?f.toLowerCase():null,b=b.toLowerCase());let c=R===b||!o&&R.startsWith(b)&&R.charAt(b.length)==="/",y=f!=null&&(f===b||!o&&f.startsWith(b)&&f.charAt(b.length)==="/"),N=c?a:void 0,P;typeof i=="function"?P=i({isActive:c,isPending:y}):P=[i,c?"active":null,y?"pending":null].filter(Boolean).join(" ");let k=typeof l=="function"?l({isActive:c,isPending:y}):l;return v.createElement(B,A({},p,{"aria-current":N,className:P,ref:r,style:k,to:s}),typeof u=="function"?u({isActive:c,isPending:y}):u)});It.displayName="NavLink";var jt=v.forwardRef((e,t)=>v.createElement(ze,A({},e,{ref:t})));jt.displayName="Form";var ze=v.forwardRef((e,t)=>{let{reloadDocument:r,replace:a,method:n=re,action:i,onSubmit:o,fetcherKey:l,routeId:s,relative:u,preventScrollReset:p}=e,d=Re(e,Ft),m=zt(l,s),g=n.toLowerCase()==="get"?"get":"post",_=$t(i,{relative:u});return v.createElement("form",A({ref:t,method:g,action:_,onSubmit:r?o:R=>{if(o&&o(R),R.defaultPrevented)return;R.preventDefault();let f=R.nativeEvent.submitter,c=f?.getAttribute("formmethod")||n;m(f||R.currentTarget,{method:c,replace:a,relative:u,preventScrollReset:p})}},d))});ze.displayName="FormImpl";function Bt(e){let{getKey:t,storageKey:r}=e;return Wt({getKey:t,storageKey:r}),null}Bt.displayName="ScrollRestoration";var ae;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmitImpl="useSubmitImpl",e.UseFetcher="useFetcher"})(ae||(ae={}));var be;(function(e){e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(be||(be={}));function $e(e){return e+" must be used within a data router.  See https://reactrouter.com/routers/picking-a-router."}function We(e){let t=v.useContext(J);return t||x(!1,$e(e)),t}function Vt(e){let t=v.useContext(Y);return t||x(!1,$e(e)),t}function Ht(e,t){let{target:r,replace:a,state:n,preventScrollReset:i,relative:o}=t===void 0?{}:t,l=Oe(),s=O(),u=G(e,{relative:o});return v.useCallback(p=>{if(At(p,r)){p.preventDefault();let d=a!==void 0?a:$(s)===$(u);l(e,{replace:d,state:n,preventScrollReset:i,relative:o})}},[s,l,u,a,n,r,e,i,o])}function zt(e,t){let{router:r}=We(ae.UseSubmitImpl),{basename:a}=v.useContext(C),n=Ie();return v.useCallback(function(i,o){if(o===void 0&&(o={}),typeof document>"u")throw new Error("You are calling submit during the server render. Try calling submit within a `useEffect` or callback instead.");let{action:l,method:s,encType:u,formData:p}=Lt(i,o,a),d={preventScrollReset:o.preventScrollReset,formData:p,formMethod:s,formEncType:u};e?(t==null&&x(!1,"No routeId available for useFetcher()"),r.fetch(e,t,l,d)):r.navigate(l,A({},d,{replace:o.replace,fromRouteId:n}))},[r,a,e,t,n])}function $t(e,t){let{relative:r}=t===void 0?{}:t,{basename:a}=v.useContext(C),n=v.useContext(F);n||x(!1,"useFormAction must be used inside a RouteContext");let[i]=n.matches.slice(-1),o=A({},G(e||".",{relative:r})),l=O();if(e==null&&(o.search=l.search,o.hash=l.hash,i.route.index)){let s=new URLSearchParams(o.search);s.delete("index"),o.search=s.toString()?"?"+s.toString():""}return(!e||e===".")&&i.route.index&&(o.search=o.search?o.search.replace(/^\?/,"?index&"):"?index"),a!=="/"&&(o.pathname=o.pathname==="/"?a:K([a,o.pathname])),$(o)}var He="react-router-scroll-positions",te={};function Wt(e){let{getKey:t,storageKey:r}=e===void 0?{}:e,{router:a}=We(ae.UseScrollRestoration),{restoreScrollPosition:n,preventScrollReset:i}=Vt(be.UseScrollRestoration),o=O(),l=Be(),s=je();v.useEffect(()=>(window.history.scrollRestoration="manual",()=>{window.history.scrollRestoration="auto"}),[]),Kt(v.useCallback(()=>{if(s.state==="idle"){let u=(t?t(o,l):null)||o.key;te[u]=window.scrollY}sessionStorage.setItem(r||He,JSON.stringify(te)),window.history.scrollRestoration="auto"},[r,t,s.state,o,l])),typeof document<"u"&&(v.useLayoutEffect(()=>{try{let u=sessionStorage.getItem(r||He);u&&(te=JSON.parse(u))}catch{}},[r]),v.useLayoutEffect(()=>{let u=a?.enableScrollRestoration(te,()=>window.scrollY,t);return()=>u&&u()},[a,t]),v.useLayoutEffect(()=>{if(n!==!1){if(typeof n=="number"){window.scrollTo(0,n);return}if(o.hash){let u=document.getElementById(o.hash.slice(1));if(u){u.scrollIntoView();return}}i!==!0&&window.scrollTo(0,0)}},[o,n,i]))}function Kt(e,t){let{capture:r}=t||{};v.useEffect(()=>{let a=r!=null?{capture:r}:void 0;return window.addEventListener("pagehide",e,a),()=>{window.removeEventListener("pagehide",e,a)}},[e,r])}function Ke(e){var t,r,a="";if(typeof e=="string"||typeof e=="number")a+=e;else if(typeof e=="object")if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(r=Ke(e[t]))&&(a&&(a+=" "),a+=r);else for(t in e)e[t]&&(a&&(a+=" "),a+=t);return a}function Jt(){for(var e,t,r=0,a="";r<arguments.length;)(e=arguments[r++])&&(t=Ke(e))&&(a&&(a+=" "),a+=t);return a}var E=Jt;import{jsx as Je}from"react/jsx-runtime";var oe=["font-semibold","block","w-full","py-2","px-5","bg-indigo-600","disabled:bg-gray-400","hover:bg-indigo-900","text-red","disabled:text-gray-200","text-center","rounded"],Yt=({className:e,to:t,...r})=>t!=null?Je(B,{to:t,className:E(e,oe),...r}):Je("button",{className:E(e,oe),...r});import{useEffect as ir}from"react";import q from"react";import{useRef as Gt}from"react";import{useState as qt,useRef as V,useCallback as Xt}from"react";import{useEffect as Qt,useLayoutEffect as Zt}from"react";var Ye=typeof window>"u"?Qt:Zt,er=({isPlaying:e,duration:t,startAt:r=0,updateInterval:a=0,onComplete:n,onUpdate:i})=>{let[o,l]=qt(r),s=V(0),u=V(r),p=V(r*-1e3),d=V(null),m=V(null),g=V(null),_=f=>{let c=f/1e3;if(m.current===null){m.current=c,d.current=requestAnimationFrame(_);return}let y=c-m.current,N=s.current+y;m.current=c,s.current=N;let P=u.current+(a===0?N:(N/a|0)*a),k=u.current+N,T=typeof t=="number"&&k>=t;l(T?t:P),T||(d.current=requestAnimationFrame(_))},b=()=>{d.current&&cancelAnimationFrame(d.current),g.current&&clearTimeout(g.current),m.current=null},R=Xt(f=>{b(),s.current=0;let c=typeof f=="number"?f:r;u.current=c,l(c),e&&(d.current=requestAnimationFrame(_))},[e,r]);return Ye(()=>{if(i?.(o),t&&o>=t){p.current+=t*1e3;let{shouldRepeat:f=!1,delay:c=0,newStartAt:y}=n?.(p.current/1e3)||{};f&&(g.current=setTimeout(()=>R(y),c*1e3))}},[o,t]),Ye(()=>(e&&(d.current=requestAnimationFrame(_)),b),[e,t,a]),{elapsedTime:o,reset:R}},tr=(e,t,r)=>{let a=e/2,n=t/2,i=a-n,o=2*i,l=r==="clockwise"?"1,0":"0,1",s=2*Math.PI*i;return{path:`m ${a},${n} a ${i},${i} 0 ${l} 0,${o} a ${i},${i} 0 ${l} 0,-${o}`,pathLength:s}},Ge=(e,t)=>e===0||e===t?0:typeof t=="number"?e-t:0,rr=e=>({position:"relative",width:e,height:e}),ar={display:"flex",justifyContent:"center",alignItems:"center",position:"absolute",left:0,top:0,width:"100%",height:"100%"},Xe=(e,t,r,a,n)=>{if(a===0)return t;let i=(n?a-e:e)/a;return t+r*i},qe=e=>{var t,r;return(r=(t=e.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i,(a,n,i,o)=>`#${n}${n}${i}${i}${o}${o}`).substring(1).match(/.{2}/g))==null?void 0:t.map(a=>parseInt(a,16)))!=null?r:[]},nr=(e,t)=>{var r;let{colors:a,colorsTime:n,isSmoothColorTransition:i=!0}=e;if(typeof a=="string")return a;let o=(r=n?.findIndex((m,g)=>m>=t&&t>=n[g+1]))!=null?r:-1;if(!n||o===-1)return a[0];if(!i)return a[o];let l=n[o]-t,s=n[o]-n[o+1],u=qe(a[o]),p=qe(a[o+1]),d=!!e.isGrowing;return`rgb(${u.map((m,g)=>Xe(l,m,p[g]-m,s,d)|0).join(",")})`},or=e=>{let{duration:t,initialRemainingTime:r,updateInterval:a,size:n=180,strokeWidth:i=12,trailStrokeWidth:o,isPlaying:l=!1,isGrowing:s=!1,rotation:u="clockwise",onComplete:p,onUpdate:d}=e,m=Gt(),g=Math.max(i,o??0),{path:_,pathLength:b}=tr(n,g,u),{elapsedTime:R}=er({isPlaying:l,duration:t,startAt:Ge(t,r),updateInterval:a,onUpdate:typeof d=="function"?c=>{let y=Math.ceil(t-c);y!==m.current&&(m.current=y,d(y))}:void 0,onComplete:typeof p=="function"?c=>{var y;let{shouldRepeat:N,delay:P,newInitialRemainingTime:k}=(y=p(c))!=null?y:{};if(N)return{shouldRepeat:N,delay:P,newStartAt:Ge(t,k)}}:void 0}),f=t-R;return{elapsedTime:R,path:_,pathLength:b,remainingTime:Math.ceil(f),rotation:u,size:n,stroke:nr(e,f),strokeDashoffset:Xe(R,0,b,t,s),strokeWidth:i}},_e=e=>{let{children:t,strokeLinecap:r,trailColor:a,trailStrokeWidth:n}=e,{path:i,pathLength:o,stroke:l,strokeDashoffset:s,remainingTime:u,elapsedTime:p,size:d,strokeWidth:m}=or(e);return q.createElement("div",{style:rr(d)},q.createElement("svg",{viewBox:`0 0 ${d} ${d}`,width:d,height:d,xmlns:"http://www.w3.org/2000/svg"},q.createElement("path",{d:i,fill:"none",stroke:a??"#d9d9d9",strokeWidth:n??m}),q.createElement("path",{d:i,fill:"none",stroke:l,strokeLinecap:r??"round",strokeWidth:m,strokeDasharray:o,strokeDashoffset:s})),typeof t=="function"&&q.createElement("div",{style:ar},t({remainingTime:u,elapsedTime:p,color:l})))};_e.displayName="CountdownCircleTimer";import{Fragment as sr,jsx as Qe}from"react/jsx-runtime";var lr=({onAfterCountdown:e,duration:t=1e3})=>(ir(()=>{let r=setTimeout(e,t);return()=>clearTimeout(r)}),Qe(_e,{isPlaying:!0,duration:t/1e3,colors:["#85a2d1","#85a2d1"],colorsTime:[1,0],children:()=>Qe(sr,{})}));import{useCallback as S,useState as cr}from"react";var Ze="4b241708b8653e8812ab94f87d01986bbb65428c6c824463420f49bb501a8813",ur=`:root {
  --dpad-bg-color: #fff;
  --dpad-bg-color-hover: #eee;
  --dpad-bg-color-active: #fff;
  --dpad-bg-color-disabled: #fff;
  --dpad-fg-color: #5217b8;
  --dpad-fg-color-hover: #5217b8;
  --dpad-fg-color-active: #ffb300;
  --dpad-fg-color-disabled: #bbb;

  --dpad-button-outer-radius: 15%;
  --dpad-button-inner-radius: 50%;

  --dpad-arrow-position: 40%;
  --dpad-arrow-position-hover: 35%;
  --dpad-arrow-base: 19px;
  --dpad-arrow-height: 13px;
}

._dpad_vt3ah_20 {
  position: relative;
  display: inline-block;

  width: 200px;
  height: 200px;

  overflow: hidden;
}

/* Buttons background */

._up_vt3ah_32,
._right_vt3ah_33,
._down_vt3ah_34,
._left_vt3ah_35 {
  display: block;
  position: absolute;
  -webkit-tap-highlight-color: rgba(255, 255, 255, 0);

  line-height: 40%;
  text-align: center;
  background: var(--dpad-bg-color);
  border-color: var(--dpad-fg-color);
  border-style: solid;
  border-width: 1px;
  padding: 0px;
  color: transparent;
}

._up_vt3ah_32,
._down_vt3ah_34 {
  width: 33.3%;
  height: 43%;
}

._left_vt3ah_35,
._right_vt3ah_33 {
  width: 43%;
  height: 33%;
}

._up_vt3ah_32 {
  top: 0;
  left: 50%;
  transform: translate(-50%, 0);
  border-radius: var(--dpad-button-outer-radius) var(--dpad-button-outer-radius) var(--dpad-button-inner-radius)
    var(--dpad-button-inner-radius);
}

._down_vt3ah_34 {
  bottom: 0;
  left: 50%;
  transform: translate(-50%, 0);
  border-radius: var(--dpad-button-inner-radius) var(--dpad-button-inner-radius) var(--dpad-button-outer-radius)
    var(--dpad-button-outer-radius);
}

._left_vt3ah_35 {
  top: 50%;
  left: 0;
  transform: translate(0, -50%);
  border-radius: var(--dpad-button-outer-radius) var(--dpad-button-inner-radius) var(--dpad-button-inner-radius)
    var(--dpad-button-outer-radius);
}

._right_vt3ah_33 {
  top: 50%;
  right: 0;
  transform: translate(0, -50%);
  border-radius: 50% var(--dpad-button-outer-radius) var(--dpad-button-outer-radius) 50%;
}

/* Buttons arrows */
._up_vt3ah_32:before,
._right_vt3ah_33:before,
._down_vt3ah_34:before,
._left_vt3ah_35:before {
  content: "";
  position: absolute;
  width: 0;
  height: 0;
  border-radius: 5px;
  border-style: solid;
  transition: all 0.25s;
}

._up_vt3ah_32:before {
  top: var(--dpad-arrow-position);
  left: 50%;
  transform: translate(-50%, -50%);
  border-width: 0 var(--dpad-arrow-height) var(--dpad-arrow-base) var(--dpad-arrow-height);
  border-color: transparent transparent var(--dpad-fg-color) transparent;
}

._down_vt3ah_34:before {
  bottom: var(--dpad-arrow-position);
  left: 50%;
  transform: translate(-50%, 50%);
  border-width: var(--dpad-arrow-base) var(--dpad-arrow-height) 0px var(--dpad-arrow-height);
  border-color: var(--dpad-fg-color) transparent transparent transparent;
}

._left_vt3ah_35:before {
  left: var(--dpad-arrow-position);
  top: 50%;
  transform: translate(-50%, -50%);
  border-width: var(--dpad-arrow-height) var(--dpad-arrow-base) var(--dpad-arrow-height) 0;
  border-color: transparent var(--dpad-fg-color) transparent transparent;
}

._right_vt3ah_33:before {
  right: var(--dpad-arrow-position);
  top: 50%;
  transform: translate(50%, -50%);
  border-width: var(--dpad-arrow-height) 0 var(--dpad-arrow-height) var(--dpad-arrow-base);
  border-color: transparent transparent transparent var(--dpad-fg-color);
}

/* Hover */

._up_vt3ah_32:hover,
._right_vt3ah_33:hover,
._down_vt3ah_34:hover,
._left_vt3ah_35:hover {
  background: var(--dpad-bg-color-hover);
  border-color: var(--dpad-fg-color-hover);
}

._up_vt3ah_32:hover:before {
  top: var(--dpad-arrow-position-hover);
  border-bottom-color: var(--dpad-fg-color-hover);
}

._down_vt3ah_34:hover:before {
  bottom: var(--dpad-arrow-position-hover);
  border-top-color: var(--dpad-fg-color-hover);
}

._left_vt3ah_35:hover:before {
  left: var(--dpad-arrow-position-hover);
  border-right-color: var(--dpad-fg-color-hover);
}

._right_vt3ah_33:hover:before {
  right: var(--dpad-arrow-position-hover);
  border-left-color: var(--dpad-fg-color-hover);
}

/* Active */

._up_vt3ah_32:active,
._right_vt3ah_33:active,
._down_vt3ah_34:active,
._left_vt3ah_35:active,
._up_vt3ah_32._active_vt3ah_175,
._right_vt3ah_33._active_vt3ah_175,
._down_vt3ah_34._active_vt3ah_175,
._left_vt3ah_35._active_vt3ah_175 {
  background: var(--dpad-bg-color-active);
  border-color: var(--dpad-fg-color-active);
}

._up_vt3ah_32:active:before,
._up_vt3ah_32._active_vt3ah_175:before {
  border-bottom-color: var(--dpad-fg-color-active);
}

._down_vt3ah_34:active:before,
._down_vt3ah_34._active_vt3ah_175:before {
  border-top-color: var(--dpad-fg-color-active);
}

._left_vt3ah_35:active:before,
._left_vt3ah_35._active_vt3ah_175:before {
  border-right-color: var(--dpad-fg-color-active);
}

._right_vt3ah_33:active:before,
._right_vt3ah_33._active_vt3ah_175:before {
  border-left-color: var(--dpad-fg-color-active);
}

/* Disabled */

._up_vt3ah_32._disabled_vt3ah_205,
._right_vt3ah_33._disabled_vt3ah_205,
._down_vt3ah_34._disabled_vt3ah_205,
._left_vt3ah_35._disabled_vt3ah_205 {
  background: var(--dpad-bg-color-disabled);
  border-color: var(--dpad-fg-color-disabled);
}

._up_vt3ah_32._disabled_vt3ah_205:before {
  top: var(--dpad-arrow-position);
  border-bottom-color: var(--dpad-fg-color-disabled);
}

._down_vt3ah_34._disabled_vt3ah_205:before {
  bottom: var(--dpad-arrow-position);
  border-top-color: var(--dpad-fg-color-disabled);
}

._left_vt3ah_35._disabled_vt3ah_205:before {
  left: var(--dpad-arrow-position);
  border-right-color: var(--dpad-fg-color-disabled);
}

._right_vt3ah_33._disabled_vt3ah_205:before {
  right: var(--dpad-arrow-position);
  border-left-color: var(--dpad-fg-color-disabled);
}
`;(function(){if(!(typeof document>"u")&&!document.getElementById(Ze)){var e=document.createElement("style");e.id=Ze,e.textContent=ur,document.head.appendChild(e)}})();var D={dpad:"_dpad_vt3ah_20",up:"_up_vt3ah_32",right:"_right_vt3ah_33",down:"_down_vt3ah_34",left:"_left_vt3ah_35",active:"_active_vt3ah_175",disabled:"_disabled_vt3ah_205"};import{jsx as ie,jsxs as hr}from"react/jsx-runtime";var w={UP:0,DOWN:1,RIGHT:2,LEFT:3},le=(e,t)=>e.find(r=>r===t)!=null,dr=()=>{let[e,t]=cr([]),r=S(a=>le(e,a),[e]);return{pressedButtons:e,isButtonPressed:r,setPressedButtons:t}},fr=({pressedButtons:e=[],onPressedButtonsChange:t=i=>{},activeButtons:r=[],disabled:a=!1,...n})=>{let i=S(c=>le(e,c),[e]),o=S(c=>le(r,c),[r]),l=S(c=>Array.isArray(a)?le(a,c):!!a,[a]),s=S(c=>{i(c)||t([...e,c])},[e,i,t]),u=S(c=>{i(c)&&t(e.filter(y=>y!==c))},[e,i,t]),p=S(()=>{s(w.DOWN)},[s]),d=S(()=>{u(w.DOWN)},[u]),m=S(()=>{s(w.UP)},[s]),g=S(()=>{u(w.UP)},[u]),_=S(()=>{s(w.LEFT)},[s]),b=S(()=>{u(w.LEFT)},[u]),R=S(()=>{s(w.RIGHT)},[s]),f=S(()=>{u(w.RIGHT)},[u]);return hr("nav",{className:D.dpad,...n,children:[ie("button",{className:E(D.up,{[D.active]:o(w.UP),[D.disabled]:l(w.UP)}),disabled:l(w.UP),onMouseDown:m,onMouseUp:g,children:"Up"}),ie("button",{className:E(D.right,{[D.active]:o(w.RIGHT),[D.disabled]:l(w.RIGHT)}),disabled:l(w.RIGHT),onMouseDown:R,onMouseUp:f,children:"Right"}),ie("button",{className:E(D.down,{[D.active]:o(w.DOWN),[D.disabled]:l(w.DOWN)}),disabled:l(w.DOWN),onMouseDown:p,onMouseUp:d,children:"Down"}),ie("button",{className:E(D.left,{[D.active]:o(w.LEFT),[D.disabled]:l(w.LEFT)}),disabled:l(w.LEFT),onMouseDown:_,onMouseUp:b,children:"Left"})]})};import"react";import{jsxs as mr}from"react/jsx-runtime";var pr=({value:e,className:t,...r})=>mr("div",{className:E(t,"text-sm","py-2","px-5","bg-slate-600","text-white","text-center","rounded-full"),...r,children:[e.toFixed(0).padStart(2,"0")," fps"]});import{useCallback as se,useEffect as yr,useState as Ee}from"react";import{useEffect as vr}from"react";var L=(e,t)=>{vr(()=>(document.addEventListener(e,t),()=>{document.removeEventListener(e,t)}),[e,t])};var et="0a1fb8ea48ce041f16ce1db72b493057f5f393a07c6ae0be3446552f34bb1118",gr=`:root {
  --joystick-surface-size: 200px;
  --joystick-stick-size: 50px;

  --joystick-color: #5217b8;
  --joystick-color-active: #ffb300;
  --joystick-color-disabled: #bbb;
}

._joystick_1jpad_10 {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;

  overflow: hidden;

  height: var(--joystick-surface-size);
  width: var(--joystick-surface-size);
  border-radius: calc(var(--joystick-stick-size) / 2);
  border-color: var(--joystick-color);
  border-width: 1px;
  border-style: solid;

  transition: all 0.25s;
}

._joystick_1jpad_10 > ._stick_1jpad_28 {
  height: var(--joystick-stick-size);
  width: var(--joystick-stick-size);

  border-radius: 50%;
  background-color: var(--joystick-color);

  transition: all 0.25s;
}

._joystick_1jpad_10._active_1jpad_38 {
  border-color: var(--joystick-color-active);
}

._joystick_1jpad_10._active_1jpad_38 > ._stick_1jpad_28 {
  background-color: var(--joystick-color-active);
}

._joystick_1jpad_10._disabled_1jpad_46 {
  border-color: var(--joystick-color-disabled);
}

._joystick_1jpad_10._disabled_1jpad_46 > ._stick_1jpad_28 {
  background-color: var(--joystick-color-disabled);
}
`;(function(){if(!(typeof document>"u")&&!document.getElementById(et)){var e=document.createElement("style");e.id=et,e.textContent=gr,document.head.appendChild(e)}})();var X={joystick:"_joystick_1jpad_10",stick:"_stick_1jpad_28",active:"_active_1jpad_38",disabled:"_disabled_1jpad_46"};import{jsx as nt}from"react/jsx-runtime";var br=200,Rr=50,tt=br/2-Rr/2,rt=(e,t,r)=>e.map((a,n)=>Math.max(Math.min(a,r[n]),t[n])),at=(e,t)=>e.map(r=>r*t),_r=()=>{let[e,t]=Ee([0,0]),[r,a]=Ee(!1),n=se((i,o=!1)=>{t(i),a(o)},[t,a]);return{joystickPosition:e,isJoystickActive:r,setJoystickState:n}},H={position:[0,0],active:!1,onChange:([e,t],r)=>{},lowerBound:[-1,-1],upperBound:[1,1],disabled:!1},Er=({position:e=H.position,active:t=H.active,onChange:r=H.onChange,lowerBound:a=H.lowerBound,upperBound:n=H.upperBound,disabled:i=H.disabled,...o})=>{let[l,s]=Ee(null);yr(()=>{i&&(s(null),r([0,0],!1))},[i,r,s]);let u=se(g=>{i||s([g.clientX,g.clientY])},[i,s]),p=se(()=>{s(null),r([0,0],!1)},[s,r]),d=se(g=>{l&&r(rt(at([g.clientX-l[0],g.clientY-l[1]],1/tt),a,n),!0)},[l,r,a,n]),m=at(rt(e,a,n),tt);return L("mousemove",d),L("mouseup",p),nt("div",{className:E(X.joystick,{[X.disabled]:i,[X.active]:t}),...o,children:nt("div",{className:E(X.stick),onMouseDown:u,style:{transform:`translate(${m[0]}px, ${m[1]}px)`}})})};import"react";import{jsx as we,jsxs as xr}from"react/jsx-runtime";var wr=({items:e,className:t,...r})=>we("ul",{className:E(t,"list-disc list-inside text-sm py-3"),children:e.filter(a=>!!a).map(([a,n],i)=>xr("li",{children:[we("span",{className:"font-semibold bg-indigo-200 lowercase",children:`${a}:`}),we("span",{className:"lowercase",children:` ${n}`})]},i))});import"react";import{jsx as Dr}from"react/jsx-runtime";var Nr=({className:e,...t})=>Dr(B,{className:E(e,oe),...t});import{useEffect as Cr,useRef as Pr}from"react";var ot="8627ee0b97350f75094b92f49619b88280df8cbf943ec122d0a6af1668f38714",Sr=`._container_wgdhb_1 {
  position: relative;
  display: grid;
  justify-content: center;
  height: 100%;
}

._canvas_wgdhb_8,
._overlay_wgdhb_9 {
  height: 75vh;
  grid-area: 1/-1;
  z-index: 0;

  width: auto;
}

._overlay_wgdhb_9 {
  z-index: 1;

  display: flex;
  justify-content: center;
  align-items: center;
}
`;(function(){if(!(typeof document>"u")&&!document.getElementById(ot)){var e=document.createElement("style");e.id=ot,e.textContent=Sr,document.head.appendChild(e)}})();var Q={container:"_container_wgdhb_1",canvas:"_canvas_wgdhb_8",overlay:"_overlay_wgdhb_9"};import{jsx as xe,jsxs as Ur}from"react/jsx-runtime";function Ar(e){let t=Array.prototype.map.call(e,function(r){return String.fromCharCode(r)}).join("");return btoa(t)}var Lr=({observation:e,overlay:t,className:r,splashScreenSrc:a,...n})=>{let i=Pr(),o=e?.overriddenPlayers!=null&&e.overriddenPlayers.length>0;return Cr(()=>{let l=i?.current;if(!l)return;let s=e?.renderedFrame;s&&(l.src="data:image/png;base64,"+Ar(s))},[i,e]),Ur("div",{className:E(Q.container,r),...n,children:[xe("img",{className:E(Q.canvas,{blur:t!=null}),ref:i,src:a,alt:"current observation rendered pixels"}),t?xe("div",{className:Q.overlay,children:t}):null,o?xe("div",{className:E(Q.overlay,"ring-inset","ring-8","ring-sky-500:80","duration-75")}):null]})};import"react";var it="8f0a992c36d9bad3210d1f3990a1b02607f83c01eabb9891c7604b5037b060c1",kr=`/* The switch - the box around the slider */
._switch_1qoun_2 {
    position: relative;
    display: inline-block;
    width: 60px;
    height: 34px;
  }
  
  /* Hide default HTML checkbox */
  ._switch_1qoun_2 input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  
  /* The slider */
  ._slider_1qoun_17 {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    -webkit-transition: .4s;
    transition: .4s;
  }
  
  ._slider_1qoun_17:before {
    position: absolute;
    content: "";
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    -webkit-transition: .4s;
    transition: .4s;
  }
  
  input:checked + ._slider_1qoun_17 {
    background-color: #2196F3;
  }
  
  input:focus + ._slider_1qoun_17 {
    box-shadow: 0 0 1px #2196F3;
  }
  
  input:checked + ._slider_1qoun_17:before {
    -webkit-transform: translateX(26px);
    -ms-transform: translateX(26px);
    transform: translateX(26px);
  }
  
  /* Rounded sliders */
  ._slider_1qoun_17._round_1qoun_56 {
    border-radius: 34px;
  }
  
  ._slider_1qoun_17._round_1qoun_56:before {
    border-radius: 50%;
  }`;(function(){if(!(typeof document>"u")&&!document.getElementById(it)){var e=document.createElement("style");e.id=it,e.textContent=kr,document.head.appendChild(e)}})();var ue={switch:"_switch_1qoun_2",slider:"_slider_1qoun_17",round:"_round_1qoun_56"};import{jsx as Ne,jsxs as lt}from"react/jsx-runtime";var Fr=({check:e,onChange:t,label:r})=>lt("div",{className:"flex items-center gap-2",children:[Ne("span",{children:r}),lt("label",{className:ue.switch,children:[Ne("input",{type:"checkbox",checked:e,onChange:t}),Ne("span",{className:E(ue.slider,ue.round)})]})]});import{Context as Or}from"@cogment/cogment-js-sdk";import{useEffect as Tr,useRef as Mr,useState as ce}from"react";var De="web_actor",Fn="teacher",st="player",On="observer",Tn="evaluator";var U={JOINING:"JOINING",ONGOING:"ONGOING",ENDED:"ENDED",ERROR:"ERROR"},Ir=(e,t,r,a=5e3)=>{let[[n,i],o]=ce([U.JOINING,null]),[l,s]=ce({observation:void 0,message:void 0,reward:void 0,last:!1,tickId:0}),[u,p]=ce(),[d,m]=ce(null),g=Mr(!1);return Tr(()=>{o([U.JOINING,null]);let _=new Or(e,"cogment_verse_web"),b=setTimeout(()=>{let f=new Error("Joined trial didn't start actor after timeout");console.error(`Error while running trial [${r}]`,f),o([U.ERROR,f])},a),R=async f=>{try{if(f.getTrialId()!==r)throw new Error(`Unexpected error, joined trial [${f.getTrialId()}] doesn't match desired trial [${r}]`);m({name:f.name,config:f.config,className:f.className}),p(()=>y=>{if(g.current){console.warn(`trial [${f.getTrialId()}] at tick [${f.getTickId()}] received a 2nd action, ignoring it.`);return}f.doAction(y),g.current=!0}),g.current=!1,f.start(),clearTimeout(b),o([U.ONGOING,null]);let c=f.getTickId();for await(let{observation:y,messages:N,rewards:P,type:k}of f.eventLoop()){let T={observation:y,message:N[0],reward:P[0],last:k===3,tickId:f.getTickId()},ct=T.tickId!==c;s(T),ct&&(g.current=!1),c=T.tickId}o([U.ENDED,null])}catch(c){throw o([U.ERROR,c]),console.error(`Error while running trial [${r}]`,c),c}};return _.registerActor(R,De,st),_.joinTrial(r,t,De).catch(f=>{o([U.ERROR,f]),console.error(`Error while running trial [${r}]`,f)}),()=>clearTimeout(b)},[e,t,r,a]),[n,d,l,u,i]};import{useCallback as Se,useState as jr}from"react";var Br=(e,t)=>{let r=Se(a=>{a.key===e&&(t(),a.stopPropagation(),a.preventDefault())},[e,t]);L("keyup",r)},Vr=()=>{let[e,t]=jr(new Set),r=Se(n=>{n.stopPropagation(),n.preventDefault(),t(i=>(i.add(n.key),i))},[t]);L("keydown",r);let a=Se(n=>{n.stopPropagation(),n.preventDefault(),t(i=>(i.delete(n.key),i))},[t]);return L("keyup",a),e};import{useEffect as Hr,useState as ut}from"react";var zr=(e,t=30,r=!0)=>{let[a,n]=ut(t),[i,o]=ut(null);return Hr(()=>{if(r)return;let l=1e3/t,s=i!=null?Math.max(0,l-new Date().getTime()+i):0,u=setTimeout(()=>{let p=new Date().getTime(),d=i!=null?new Date().getTime()-i:l;e(d),o(p),n(1e3/d)},s);return()=>{clearTimeout(u)}},[r,t,e,i,o,n]),{currentFps:a}};export{Yt as Button,lr as Countdown,w as DPAD_BUTTONS,fr as DPad,Tn as EVALUATOR_ACTOR_CLASS,pr as FpsCounter,Er as Joystick,wr as KeyboardControlList,Nr as Link,On as OBSERVER_ACTOR_CLASS,st as PLAYER_ACTOR_CLASS,Lr as RenderedScreen,Fr as Switch,Fn as TEACHER_ACTOR_CLASS,U as TRIAL_STATUS,De as WEB_ACTOR_NAME,dr as useDPadPressedButtons,L as useDocumentEventListener,Br as useDocumentKeypressListener,Ir as useJoinedTrial,_r as useJoystickState,Vr as usePressedKeys,zr as useRealTimeUpdate};
/*! Bundled license information:

@remix-run/router/dist/router.js:
  (**
   * @remix-run/router v1.6.2
   *
   * Copyright (c) Remix Software Inc.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE.md file in the root directory of this source tree.
   *
   * @license MIT
   *)

react-router/dist/index.js:
  (**
   * React Router v6.11.2
   *
   * Copyright (c) Remix Software Inc.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE.md file in the root directory of this source tree.
   *
   * @license MIT
   *)

react-router-dom/dist/index.js:
  (**
   * React Router DOM v6.11.2
   *
   * Copyright (c) Remix Software Inc.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE.md file in the root directory of this source tree.
   *
   * @license MIT
   *)
*/
//# sourceMappingURL=index.js.map
