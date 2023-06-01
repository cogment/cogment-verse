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

import React, { useEffect, useState } from "react";

export const DynamicallyImportedComponent = ({ moduleUrl, componentName = "default", ...otherProps }) => {
  const [error, setError] = useState(null);
  const [Component, setComponent] = useState(null);

  useEffect(() => {
    setComponent(null);
    setError(null);
    let canceled = false;
    import(moduleUrl)
      .then((module) => {
        if (canceled) {
          return;
        }
        const Component = module[componentName];
        if (Component == null) {
          setError(new Error(`"${componentName}" not found in "${moduleUrl}"`));
          return;
        }
        setComponent(() => Component);
      })
      .catch((error) => {
        if (!canceled) {
          setError(error);
        }
      });
    return () => {
      canceled = true;
      setComponent(null);
      setError(null);
    };
  }, [moduleUrl]);

  if (error != null) {
    throw error;
  }

  if (Component != null) {
    return <Component {...otherProps} />;
  }

  return <div>importing {moduleUrl}...</div>;
};
