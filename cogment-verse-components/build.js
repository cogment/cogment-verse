const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");
const pkg = require("./package.json");

esbuild
  .build({
    entryPoints: ["./src/index.js"],
    target: "es2020",
    format: "esm",
    bundle: true,
    minify: false,
    sourcemap: true,
    outdir: "dist",
    plugins: [cssModulesPlugin()],
    external: Object.keys(pkg.peerDependencies),
  })
  .catch(() => process.exit(1));
