// usage: node make_m2.js out.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, ShadingType, HeadingLevel, Footer, PageNumber, TabStopType, LevelFormat,
} = require("docx");

const SERIF = "Times New Roman", SANS = "Arial";
const NAVY = "000000", GREY = "595959";

// body text: `code` -> Courier New, **bold** -> bold
function runs(text) {
  const out = [];
  text.split("`").forEach((s, i) => {
    if (i % 2) { out.push(new TextRun({ text: s, font: SERIF, size: 21 })); return; }  // code names in the body font
    s.split("**").forEach((p, j) => {
      // [[...]] marks text the group still has to confirm
      p.split(/\[\[|\]\]/).forEach((q, k) => {
        if (q) out.push(new TextRun({ text: q, font: SERIF, size: 21, bold: j % 2 === 1,
          highlight: k % 2 ? "yellow" : undefined }));
      });
    });
  });
  return out;
}
const body = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 50, line: 240 },
  children: runs(t) });
const heading = (t, before = 90) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true,
  spacing: { before, after: 40 },
  children: [new TextRun({ text: t, bold: true, size: 22, font: SANS, color: NAVY })] });
const hint = (t) => new Paragraph({ spacing: { after: 60 },
  shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
  children: [new TextRun({ text: t, italics: true, size: 17, font: SANS, color: "7F7F7F" })] });
const caption = (label, t, before, after, keepNext = false) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before, after }, keepNext,
  children: [new TextRun({ text: label + " ", bold: true, size: 17, font: SANS }),
    new TextRun({ text: t, size: 17, font: SANS, color: "404040" })] });

// a block of "- " lines is a bullet list; a block that is only bold text heads the lines below it
const bullet = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, numbering: { reference: "dots", level: 0 },
  spacing: { after: 20, line: 240 }, children: runs(t) });
const group = (t) => new Paragraph({ keepNext: true, spacing: { before: 40, after: 20 }, children: runs(t) });
const block = (para) => (t) => t === "{{fig1}}" ? figure1() : t === "{{table1}}" ? table1() : t === "{{fig2}}" ? figure2()
  : t.startsWith("- ") ? t.split(/\n(?=- )/).map((b) => bullet(b.slice(2).replace(/\n/g, " ")))
  : /^\*\*[^*]+\*\*( \([^)]*\))?$/.test(t) ? [group(t)] : [para(t)];
const section = (n, para = body) => fs.readFileSync(`${__dirname}/sec${n}.md`, "utf8").trim().split(/\n\s*\n/).flatMap(block(para));

// table 1: three-line table across the page, on-time and served rates side by side; one-cell rows head a group
const thick = { style: BorderStyle.SINGLE, size: 10, color: "000000" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const W = [3466, 1050, 1100, 1050, 1050, 1100, 1050];
const TW = W.reduce((a, b) => a + b, 0);
const rows = [
  ["Random", "55.5", "26.3", "22.6", "84.4", "52.3", "44.1"],
  ["EDF (earliest deadline first)", "99.6", "40.6", "27.5", "100.0", "89.0", "72.1"],
  ["FCFS (first come, first served)", "97.0", "49.3", "32.1", "99.9", "84.6", "71.4"],
  ["Nearest-first (soonest to finish)", "99.2", "71.6", "53.2", "100.0", "93.2", "83.7"],
  ["Slack-aware (main teacher)", "99.3", "81.0", "65.9", "100.0", "90.6", "80.7"],
  ["Rollout planner (slow look-ahead)", "99.9", "89.7", "76.8", "100.0", "94.6", "85.4"],
  ["Imitating Slack-aware"],
  ["MLP", "99.2", "80.2", "63.7", "100.0", "91.5", "80.5"],
  ["Set network", "99.3", "81.2", "66.2", "100.0", "91.3", "80.5"],
  ["Imitating rollout"],
  ["MLP", "98.8", "82.5", "67.6", "100.0", "95.1", "85.7"],
  ["Set network", "99.3", "84.1", "72.6", "99.9", "93.3", "85.6"]];
const GROUP = rows.findIndex((r) => r.length === 1);  // first group row: a rule above, rows below indented
const tcell = (t, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA }, columnSpan: o.span || 1,
  borders: { left: none, right: none, top: o.top || none, bottom: o.bottom || none },
  margins: { top: 2, bottom: 2, left: o.indent ? 180 : 60, right: 40 },
  children: [new Paragraph({ alignment: o.left ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: !o.last,
    children: [new TextRun({ text: t, size: 16, font: SANS, bold: o.bold, italics: o.italics })] })] });
const head = [
  new TableRow({ cantSplit: true, children: [tcell("", W[0], { top: thick }),
    tcell("On-time rate (%)", W[1] + W[2] + W[3], { span: 3, top: thick, bottom: thin, bold: true }),
    tcell("Served rate (%)", W[4] + W[5] + W[6], { span: 3, top: thick, bottom: thin, bold: true })] }),
  new TableRow({ cantSplit: true, children: ["Policy", "Low (22)", "Medium (52)", "High (68)", "Low", "Medium", "High"]
    .map((t, i) => tcell(t, W[i], { bottom: thin, bold: true, left: i === 0 })) })];
const tableRows = rows.map((r, k) => {
  const last = k === rows.length - 1, o = { top: k === GROUP ? thin : none, bottom: last ? thick : none, last };
  return new TableRow({ cantSplit: true, children: r.length === 1
    ? [tcell(r[0], TW, { ...o, span: 7, left: true, italics: true })]
    : r.map((t, i) => tcell(t, W[i], { ...o, left: i === 0, indent: i === 0 && k > GROUP })) });
});
const table1 = () => [
  caption("Table 1.", "Mean daily on-time and served rates (orders delivered on time, or at all, as % of orders released) at each demand level, map 0, the same 40 development days; orders per day in brackets; networks trained once.", 60, 30, true),
  new Table({ alignment: AlignmentType.CENTER, width: { size: TW, type: WidthType.DXA }, columnWidths: W,
    rows: [...head, ...tableRows] })];

const figure1 = () => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(`${__dirname}/setnet.png`),
      transformation: { width: 510, height: 148 } })] }),
  caption("Figure 1.", "The set network from observation to output (B: batch size; ⊕: concatenation). Encoders use ReLU throughout; each head has one ReLU hidden layer of 64 and a linear output.", 40, 100)];
const figure2 = () => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(`${__dirname}/bc_curve.png`),
      transformation: { width: 470, height: 115 } })] }),
  caption("Figure 2.", "Validation decisions matching the teacher (%), one point per epoch; ○ marks the epoch kept (lowest validation loss).", 20, 80)];

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Dynamic Medical Pickup and Delivery with a Single UAV",
      bold: true, size: 32, font: SANS, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
    children: [new TextRun({ text: "Group 17: Jihang Cai, Yu Liu, Yufeng Zhang", size: 20, font: SANS })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 6 } },
    children: [new TextRun({ text: "NUS ME5418  |  Milestone 2: Neural Network Code and Report",
      size: 18, font: SANS, color: GREY })] }),

  body("Our agent dispatches one UAV that delivers medical orders between a medical center and five hospitals under deadline, battery and payload limits. M2 builds the policy and value networks (our proposal's set/attention network and an MLP control), tests by imitation that they can learn, and updates the environment, baselines and visualization."),
  heading("1. Network code"), ...section(1),
  heading("2. Changes to the environment and evaluation"), ...section(2),
  heading("3. Training by imitation"), ...section(3),
  heading("4. Demo and visualization"), ...section(4),
  heading("5. Existing code, libraries and AI use"), ...section(5),
  heading("6. Reflections and lessons learned"), ...section(6),
];

const doc = new Document({
  numbering: { config: [{ reference: "dots", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 300, hanging: 200 } } } }] }] },
  styles: { default: { document: { run: { font: SERIF, size: 21 } } },
    paragraphStyles: [{ id: "Footer", name: "footer", run: { font: SANS, size: 16, color: "808080" } }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 900, bottom: 900, left: 1020, right: 1020, footer: 450 } } },
    footers: { default: new Footer({ children: [new Paragraph({ style: "Footer",
      tabStops: [{ type: TabStopType.RIGHT, position: 9866 }],
      border: { top: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 4 } },
      children: [new TextRun({ text: "ME5418 Group 17  |  Milestone 2 report", size: 16, color: "808080", font: SANS }),
        new TextRun({ text: "\tPage ", size: 16, color: "808080", font: SANS }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080", font: SANS })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => fs.writeFileSync(process.argv[2], b));
