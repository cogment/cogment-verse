import{useCallback as P,useState as A}from"react";import{useDocumentKeypressListener as D,usePressedKeys as L,PlayObserver as O,useRealTimeUpdate as S,TEACHER_ACTOR_CLASS as _,OBSERVER_ACTOR_CLASS as v,DPad as g,useDPadPressedButtons as h,DPAD_BUTTONS as s,Button as U,FpsCounter as y,KeyboardControlList as x,serializePlayerAction as o,TEACHER_NOOP_ACTION as F,Space as I,SimplePlay as H}from"@cogment/cogment-verse";import{jsx as r,jsxs as m}from"react/jsx-runtime";var b=[],K=[s.UP],u=new I({discrete:{n:4}}),V=({sendAction:e,fps:l=20,actorParams:d,...p})=>{let i=d?.className===_,[c,f]=A(!1),T=P(()=>f(C=>!C),[f]);D("p",T);let a=L(),{pressedButtons:N,isButtonPressed:n,setPressedButtons:E}=h(),[B,t]=A([]),R=P(C=>{if(a.has("ArrowRight")||n(s.RIGHT)){t([s.RIGHT]),e(o(u,1));return}else if(a.has("ArrowDown")||n(s.DOWN)){t([s.DOWN]),e(o(u,2));return}else if(a.has("ArrowLeft")||n(s.LEFT)){t([s.LEFT]),e(o(u,3));return}else if(i){if(a.has("ArrowUp")||n(s.UP)){t([s.UP]),e(o(u,0));return}t([]),e(F);return}t([]),e(o(u,0))},[n,a,e,t,i]),{currentFps:w}=S(R,l,c);return m("div",{...p,children:[r("div",{className:"flex flex-row p-5 justify-center",children:r(g,{pressedButtons:N,onPressedButtonsChange:E,activeButtons:B,disabled:c||(i?b:K)})}),m("div",{className:"flex flex-row gap-1",children:[r(U,{className:"flex-1",onClick:T,children:c?"Resume":"Pause"}),r(y,{className:"flex-none w-fit",value:w})]}),r(x,{items:[["Left/Right Arrows","Fire left/right engine"],["Down Arrow","Fire the main engine"],i?["Up Arrow","turn off engine"]:null,["p","Pause/Unpause"]]})]})},k=({actorParams:e,...l})=>e?.className===v?r(O,{actorParams:e,...l}):r(H,{actorParams:e,...l,controls:V}),j=k;export{j as default};
//# sourceMappingURL=LunarLander.js.map
