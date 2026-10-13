// Dependency-free DOM probe for stage control, candidate linkage, and point IDs.
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, value: '', checked: id === 'showPoints' || id === 'showCandidates',
    clientWidth: 500, clientHeight: 400, innerHTML: '', textContent: '', handlers: {}, style: {},
    classList: {add() {}, remove() {}},
    addEventListener(name, fn) { this.handlers[name] = fn; },
    querySelectorAll() { return []; },
    setAttribute() {}, append() {}, getBoundingClientRect() { return {left: 0, top: 0, width: 500, height: 400}; },
  });
  return elements.get(id);
}
const draw = {setTransform() {}, fillRect() {}, beginPath() {}, arc() {}, fill() {}};
element('cloud').getContext = () => draw;
element('cloud').setPointerCapture = () => {};
const document = {getElementById: element, createElementNS: () => element('svgchild')};
const fetch = async path => {
  let payload = {shape: [1, 1], values: [[[1]]]};
  if (path === '/api/index') payload = {traces: []};
  if (path.includes('/api/details')) payload = {point_id: 17, l0_voxel: 4, slot: 2};
  return {ok: true, status: 200, headers: {get: () => 'application/json'},
    text: async () => JSON.stringify(payload)};
};
const sandbox = {document, fetch, console, devicePixelRatio: 1, ResizeObserver: class {observe() {}},
  setInterval: () => 1, clearInterval() {}, Math, Number, String, Object, Array, Promise};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
vm.runInContext(`data={sample:{m_V:true,image_view_wh:[100,100],image_source_wh:[100,100],gt3d_valid:false,
  gt_box_xyxy_px:null,image_path:'real.png',query_time:1,image_time:1},points:[[0,0,0]],point_ids:[17],
  sensor_id:[0],delta_t:[-0.1],voxel_centers:[],voxel_ids:[],tensor_stats:{},selector_top:[],
  candidates:[{row:0,type:1,xyz:[1,0,0],box:null,score3d_before:.4,score3d_after:.5,
  projected_uv:[50,50],projection_valid:true}],attention:{cross_modal:'unavailable'},flags:{}};
  events={association:{pairs:[]}};`, sandbox);
element('query').value = 'sample_1'; element('model').value = 'A2';
element('stageRange').value = '9';
vm.runInContext('stage()', sandbox);
if (!element('stageName').textContent.includes('TopK / NMS')) throw Error('stage switch failed');
vm.runInContext('selectCandidate(0)', sandbox);
if (!element('detail').textContent.includes('"row": 0')) throw Error('candidate selection failed');
if (!element('image').innerHTML.includes('left_image.png')) throw Error('image linkage failed');
element('showVoxels').checked = false;
const canvas = element('cloud');
const p = vm.runInContext('project([0,0,0],500,400)', sandbox);
const event = {pointerId: 1, button: 0, clientX: p[0], clientY: p[1], preventDefault() {}};
canvas.handlers.pointerdown(event);
Promise.resolve(canvas.handlers.pointerup(event)).then(() => {
  if (!element('detail').textContent.includes('"point_id": 17')) throw Error('point ID linkage failed');
  console.log('viewer stage/candidate/point linkage PASS');
});
