const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");
const pkg = require("./package.json");

const NODE_ENV = process.env.NODE_ENV || "production";

esbuild
  .build({
    logLevel: "info",
    entryPoints: ["./GymLunarLander.jsx"],
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
