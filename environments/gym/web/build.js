// Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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

const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");
const pkg = require("./package.json");

const NODE_ENV = process.env.NODE_ENV || "production";

esbuild
  .build({
    logLevel: "info",
    entryPoints: [
      "./GymLunarLander.jsx",
      "./GymCartPole.jsx",
      "./GymLunarLanderContinuous.jsx",
      "./GymMountainCar.jsx",
    ],
    target: "es2020",
    format: "esm",
    bundle: true,
    minify: true,
    sourcemap: true,
    metafile: true,
    jsx: "automatic",
    outdir: "dist",
    plugins: [cssModulesPlugin()],
    external: [...Object.keys(pkg.peerDependencies), "@cogment/cogment-verse"],
    define: {
      "process.env.NODE_ENV": `"${NODE_ENV}"`,
    },
  })
  // .then((result) =>
  //   esbuild.analyzeMetafile(result.metafile, {
  //     verbose: true,
  //   })
  // )
  // .then((analysisResult) => console.log(analysisResult))
  .catch(() => process.exit(1));
