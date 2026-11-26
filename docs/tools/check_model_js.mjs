// Checks that the JavaScript PortiloopModel (docs/assets/js/portiloop.js) reproduces the
// outputs of the PyTorch PortiloopNetwork recorded by docs/tools/export_model_json.py.
//   node docs/tools/check_model_js.mjs
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const PL = require(join(here, "../assets/js/portiloop.js"));
const json = JSON.parse(readFileSync(join(here, "../assets/models/portiloop_spindle_v4_0.json"), "utf8"));

const model = new PL.PortiloopModel(json);
const ref = json.reference_test;
let h = model.zeroState();
let maxErr = 0, maxHErr = 0;
ref.windows.forEach((w, i) => {
  const r = model.step(w, h);
  h = r.h;
  maxErr = Math.max(maxErr, Math.abs(r.y - ref.outputs[i]));
  ref.hidden_states[i].forEach((v, j) => { maxHErr = Math.max(maxHErr, Math.abs(v - h[j])); });
  console.log(`step ${i}: js ${r.y.toFixed(6)}  torch ${ref.outputs[i].toFixed(6)}`);
});
console.log(`max |output error| = ${maxErr.toExponential(2)}, max |hidden error| = ${maxHErr.toExponential(2)}`);
if (maxErr > 1e-4 || maxHErr > 1e-4) { console.error("MISMATCH"); process.exit(1); }
console.log(`OK: ${model.nParameters} parameters, window ${model.windowSize} samples`);
