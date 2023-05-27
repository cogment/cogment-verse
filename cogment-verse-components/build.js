const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");

const NODE_ENV = process.env.NODE_ENV || "development";
esbuild
  .build({
    entryPoints: ["./src/index.js"],
    target: "es2020",
    format: "esm",
    bundle: true,
    minify: true,
    sourcemap: true,
    outdir: "dist",
    plugins: [cssModulesPlugin()],
  })
  .catch(() => process.exit(1));
