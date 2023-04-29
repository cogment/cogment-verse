const path = require("path");
const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");

filePath = process.argv[2];
fileBasename = path.parse(filePath).name;

esbuild
  .build({
    logLevel: "info",
    entryPoints: [filePath],
    target: "es2020",
    bundle: true,
    minify: true,
    sourcemap: true,
    outfile: `dist/${fileBasename}.js`,
    plugins: [cssModulesPlugin()],
  })
  .catch(() => process.exit(1));
