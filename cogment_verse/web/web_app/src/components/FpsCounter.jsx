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

import classNames from "classnames";

export const FpsCounter = ({ value, className, ...props }) => {
  return (
    <div
      className={classNames(
        className,
        "text-sm",
        "py-2",
        "px-5",
        "bg-slate-600",
        "text-white",
        "text-center",
        "rounded-full"
      )}
      {...props}
    >
      {value.toFixed(0).padStart(2, "0")} fps
    </div>
  );
};
