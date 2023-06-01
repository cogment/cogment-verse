const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");

const pkg = require("./package.json");

const NODE_ENV = process.env.NODE_ENV || "production";

esbuild
  .build({
    logLevel: "info",
    entryPoints: [
      "./src/react/index.js",
      "./src/react/jsx-runtime/index.js",
      "./src/react-dom/client/index.js",
      "./src/shared/index.js",
      "./src/app/index.jsx",
    ],
    outbase: "./src/",
    entryNames: "[dir]",
    target: "es2020",
    format: "esm",
    splitting: true,
    bundle: true,
    minify: true,
    sourcemap: true,
    metafile: true,
    outdir: `dist`,
    plugins: [cssModulesPlugin()],
    external: Object.keys(pkg.peerDependencies || {}),
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
