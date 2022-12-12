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

import { Link } from "react-router-dom";
import classNames from "classnames";

export const buttonClasses = [
  "font-semibold",
  "block",
  "w-full",
  "py-2",
  "px-5",
  "bg-indigo-600",
  "disabled:bg-gray-400",
  "hover:bg-indigo-900",
  "text-white",
  "disabled:text-gray-200",
  "text-center",
  "rounded",
];

export const Button = ({ className, to, ...props }) => {
  if (to != null) {
    return <Link to={to} className={classNames(className, buttonClasses)} {...props} />;
  }
  return <button className={classNames(className, buttonClasses)} {...props} />;
};
