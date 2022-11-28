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

import { useRouteError } from "react-router-dom";
import { Button } from "../components/Button";

const Error = () => {
  const error = useRouteError();
  console.error(error);
  return (
    <div>
      <h1 className="text-xl font-semibold">Oups...</h1>
      <p>{error.message}</p>
      {error.cause ? (
        <p>
          <span className="font-semibold">cause: </span>
          {error.cause.message}
        </p>
      ) : null}
      <Button to="/" reloadDocument>
        Retry
      </Button>
    </div>
  );
};

export default Error;
