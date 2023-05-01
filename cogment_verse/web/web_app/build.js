const esbuild = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");

const NODE_ENV = process.env.NODE_ENV || "development";
esbuild
  .build({
    logLevel: "info",
    entryPoints: ["./src/index.jsx"],
    target: "es2020",
    format: "esm",
    bundle: true,
    minify: true,
    sourcemap: true,
    outfile: `build/static/app.js`,
    plugins: [cssModulesPlugin()],
    define: {
      "process.env.NODE_ENV": `"${NODE_ENV}"`,
    },
  })
  .catch(() => process.exit(1));
