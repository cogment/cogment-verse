import"./chunk-ON5OQYWL.js";import{useCallback as f,useState as m}from"react";import{Button as g,DPAD_BUTTONS as s,useDPadPressedButtons as w,FpsCounter as R,KeyboardControlList as O,serializePlayerAction as l,Space as h,OBSERVER_ACTOR_CLASS as y,useDocumentKeypressListener as D,usePressedKeys as E,useRealTimeUpdate as L,SimplePlay as _,PlayObserver as x}from"@cogment/cogment-verse";import{jsx as t,jsxs as C}from"react/jsx-runtime";var n=new h({discrete:{n:5}}),U=({sendAction:e,fps:u=40,actorClass:i,observation:F,...T})=>{console.log("actorClass: "+i);let[c,p]=m(!1),d=f(()=>p(P=>!P),[p]);D("p",d);let r=E(),{pressedButtons:A,isButtonPressed:a,setPressedButtons:S}=w(),[v,o]=m([]),B=f(P=>{if(r.has("ArrowLeft")||a(s.LEFT)){o([s.LEFT]),e(l(n,1));return}else if(r.has("ArrowRight")||a(s.RIGHT)){o([s.RIGHT]),e(l(n,2));return}else if(r.has("ArrowDown")||a(s.DOWN)){o([s.DOWN]),e(l(n,3));return}else if(r.has("ArrowUp")||a(s.UP)){o([s.UP]),e(l(n,4));return}o([]),e(l(n,0))},[a,r,e,o,isTeacher]),{currentFps:N}=L(B,u,c);return C("div",{...T,children:[t("div",{className:"flex flex-row p-5 justify-center",children:t(DPad,{pressedButtons:A,onPressedButtonsChange:S,activeButtons:v,disabled:c})}),C("div",{className:"flex flex-row gap-1",children:[t(g,{className:"flex-1",onClick:d,children:c?"Resume":"Pause"}),t(R,{className:"flex-none w-fit",value:N})]}),t(O,{items:[["Left/Right/Up/Down Arrows","Move left/right/up/down"],["p","Pause/Unpause"]]})]})},b=({actorParams:e,...u})=>{let i=e?.className;return console.log("PlaySimpleTag | actorClassName: "+i),i===y?t(x,{actorParams:e,...u}):t(_,{actorParams:e,...u,controls:U})},z=b;export{U as SimpleTagControls,z as default};
//# sourceMappingURL=SimpleTag.js.map
