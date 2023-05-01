const path = require("path");
const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");

const NODE_ENV = process.env.NODE_ENV || "development";

// const filePath = process.argv[2];
// const fileBasename = path.parse(filePath).name;

esbuild
  .build({
    logLevel: "info",
    entryPoints: ["index.jsx"],
    target: "es2020",
    format: "esm",
    bundle: true,
    minify: true,
    sourcemap: true,
    outfile: `dist/index.js`,
    plugins: [cssModulesPlugin()],
    define: {
      "process.env.NODE_ENV": `"${NODE_ENV}"`,
    },
  })
  .catch(() => process.exit(1));
