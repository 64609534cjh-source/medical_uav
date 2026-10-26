// usage: node make_m2.js out.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, ShadingType, HeadingLevel, Footer, PageNumber, TabStopType,
} = require("docx");

const SERIF = "Times New Roman", SANS = "Arial";
const NAVY = "000000", GREY = "595959";

// body text: `code` -> Courier New, **bold** -> bold
function runs(text) {
  const out = [];
  text.split("`").forEach((s, i) => {
    if (i % 2) { out.push(new TextRun({ text: s, font: SERIF, size: 22 })); return; }  // code names in the body font
    s.split("**").forEach((p, j) => {
      // [[...]] marks text the group still has to confirm
      p.split(/\[\[|\]\]/).forEach((q, k) => {
        if (q) out.push(new TextRun({ text: q, font: SERIF, size: 22, bold: j % 2 === 1,
          highlight: k % 2 ? "yellow" : undefined }));
      });
    });
  });
  return out;
}
const body = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 240 },
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

const section = (n) => fs.readFileSync(`${__dirname}/sec${n}.md`, "utf8").trim().split(/\n\s*\n/).map(body);

// table 1: three-line table
const thick = { style: BorderStyle.SINGLE, size: 10, color: "000000" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const W = [3300, 1800, 1800, 1800];
const rows = [["Policy", "Low load (22)", "Medium load (52)", "High load (68)"],
  ["Random", "55.5%", "26.3%", "22.6%"], ["EDF", "99.6%", "40.6%", "27.5%"],
  ["FCFS", "97.0%", "49.3%", "32.1%"], ["Nearest-first", "99.2%", "71.6%", "53.2%"],
  ["Slack-aware (teacher)", "99.3%", "81.0%", "65.9%"],
  ["MLP, imitation (one seed)", "99.2%", "79.3%", "62.7%"], ["Set network, imitation (one seed)", "99.5%", "79.5%", "64.4%"]];
const cell = (t, i, k) => new TableCell({ width: { size: W[i], type: WidthType.DXA },
  borders: { left: none, right: none, top: k === 0 ? thick : k === 6 ? thin : none,
    bottom: k === 0 ? thin : k === rows.length - 1 ? thick : none },
  margins: { top: 20, bottom: 20, left: 80, right: 80 },
  children: [new Paragraph({ alignment: i ? AlignmentType.CENTER : AlignmentType.LEFT,
    keepNext: k < rows.length - 1,
    children: [new TextRun({ text: t, bold: k === 0, size: 18, font: SANS })] })] });
const table = new Table({ alignment: AlignmentType.CENTER, width: { size: 8700, type: WidthType.DXA },
  columnWidths: W, rows: rows.map((r, k) => new TableRow({ cantSplit: true,
    children: r.map((t, i) => cell(t, i, k)) })) });

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

  body("This milestone delivers the policy and value networks of our dispatching agent: the set/attention network described in our proposal and an MLP control with the same interface. To check that they can represent good decisions before reinforcement learning, we trained both by imitating our strongest rule. We also revised the environment's reward and observation, added two baselines and a shared evaluator, and rebuilt the visualization following the M1 feedback."),

  heading("1. Network code"), section(1)[0],
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(`${__dirname}/setnet.png`),
      transformation: { width: 650, height: 106 } })] }),
  caption("Figure 1.", "Set network: shared encoders for order and cargo slots, attention from the UAV state, a shared PICK scorer, and heads for the other actions and the value.", 40, 100),
  section(1)[1],
  heading("2. Training by imitation"), ...section(2),
  caption("Table 1.", "Mean daily on-time rate (on-time / released), map 0, the same 40 development days; mean orders/day in brackets.", 80, 40, true),
  table,
  heading("3. Changes to the environment and evaluation", 200), ...section(3),
  heading("4. Demo and visualization"),
  new Table({ width: { size: 9866, type: WidthType.DXA }, columnWidths: [4766, 5100],
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: [new TableRow({ cantSplit: true, children: [
      new TableCell({ width: { size: 4766, type: WidthType.DXA }, borders: { top: none, bottom: none, left: none, right: none },
        margins: { top: 0, bottom: 0, left: 0, right: 160 }, children: section(4) }),
      new TableCell({ width: { size: 5100, type: WidthType.DXA }, borders: { top: none, bottom: none, left: none, right: none },
        margins: { top: 0, bottom: 0, left: 120, right: 0 }, children: [
          new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 20 },
            children: [new ImageRun({ type: "png", data: fs.readFileSync(`${__dirname}/frame.png`),
              transformation: { width: 320, height: 176 } })] }),
          caption("Figure 2.", "Explanation video, decision 38 at 10:58: the set network trained by imitation takes the sample on board back to M.", 40, 60)] }),
    ] })] }),
  heading("5. Existing code, libraries and AI use"), ...section(5),
  heading("6. Reflections and lessons learned"), ...section(6),
];

const doc = new Document({
  styles: { default: { document: { run: { font: SERIF, size: 22 } } },
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
