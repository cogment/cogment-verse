// Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React, { useEffect, useRef, useState } from "react";

const DynamicModuleComponent = ({ module, ...props }) => {
  const hostRef = useRef(null);
  const [render, setRender] = useState(null);
  useEffect(() => {
    if (hostRef.current == null) {
      return;
    }
    const { mount } = module;
    const { render, unmount } = mount(hostRef.current);
    // When provided with a function setState consider them an updater, because `render` is a function we need to wrap it in an updater.
    setRender(() => render);
    return () => {
      setRender(null);
      unmount();
    };
  }, [module, hostRef]);
  useEffect(() => {
    if (render != null) {
      render(props);
    }
  }, [render, props]);

  return <div ref={hostRef} />;
};

export const createDynamicModuleComponent = (moduleUrl) => (props) => {
  const [error, setError] = useState(null);
  const [module, setModule] = useState(null);

  useEffect(() => {
    setModule(null);
    setError(null);
    let canceled = false;
    import(moduleUrl)
      .then((module) => {
        if (!canceled) {
          setModule(module);
        }
      })
      .catch((error) => {
        if (!canceled) {
          setError(error);
        }
      });
    return () => {
      canceled = true;
      setModule(null);
      setError(null);
    };
  }, [moduleUrl]);

  if (error != null) {
    throw error;
  }

  if (module != null) {
    return <DynamicModuleComponent module={module} {...props} />;
  }

  return <div>importing {moduleUrl}...</div>;
};
