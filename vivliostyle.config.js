// import { VFM } from "@vivliostyle/vfm";

const isPrint = process.argv.includes("print.pdf");

const def = {
  title: "vivliostyle-sample", // populated into `publication.json`, default to `title` of the first entry or `name` in `package.json`.
  author: "oyakata <oyakata2438@gmail.com>", // default to `author` in `package.json` or undefined.
  language: "ja", // default to undefined.
  size: "JIS-B5", // JIS-B5: 教科書サイズ、A5: 最近流行りの小さいサイズの技術書
  theme: [
    "./fonts",
    "./theme-nice-techbook",
  ],
  entry: [
    // 表紙
    // { rel: "cover" },

    // 扉
    // "00-title.md",
    "title.html",
    // 前書き
    "01-preface.md",

    // 目次
    { rel: "contents" },

    // 第一部 とにかく楽をする
    "part-easy.md",
	"chap-okazunomoto.md",


    // 第二部 Vivliostyleについて
    "part-tips.md",
	"chap-oyakata-cookingenginerring.md",
	"chap-rhodium-thermal.md",
    "chap-gomana2_1.md",
    "chap-yumechi-1-nukazuke.md",
    "chap-yumechi-2-coffee.md",
    "chap-yumechi-3-protein.md",
    "chap-mottox2-coffee.md",
    "chap-mottox2-no-cooking.md",
    "chap-forte_1_short-diet.md",
    "chap-forte_2_Seasonal-foods.md",
    "chap-forte_3_Specialty-store.md",
    "chap-forte_4_low-salt.md",
    "chap-forte_5_media.md",

    "chap-recommended_cooking_equipment.md",
    "chap-recommended_shop.md",

    "chap-ditflame_01_zosui.md",
    "chap-ditflame_02_thanko_ricecooker.md",
    "chap-ditflame_05_auto_washer.md",
    "chap-ditflame_04_cream_pasta.md",
    "chap-ditflame_06_easy_fukuromen.md",

    "chap-yuusukesan18.md",

    "chap-ponyoxa.md",

    "chap-kouno.md",
	"chap-allergy.md",
	"chap-oyakata-supermarktet.md",
	"chap-oyakata-onecup.md",
    "chap-llminatoll_1_rusk.md",
    "chap-llminatoll_2_sake.md",
	"chap-oyakata-himono.md",
	"chap-kawahara-simple.md",

    "chap-column.md",

    // 後書き
    "90-postscript.md",
    "98-authors.md",
    "99-colophon.md",
  ],
  entryContext: "./src",

  // output: [ // path to generate draft file(s). default to '{title}.pdf'
  //   './output.pdf', // the output format will be inferred from the name.
  //   {
  //     path: './book',
  //     format: 'webpub',
  //   },
  // ],
  workspaceDir: ".vivliostyle", // directory which is saved intermediate files.
  toc: {
    title: "目次", // title of table of contents. default to 'Contents'.
    sectionDepth: 1,
    includeCover: false, // include cover page in table of contents. default to 'false'.
  },
  // cover: './cover.png', // cover image. default to undefined.
  vfm: {
    // options of VFM processor
    //   hardLineBreaks: true, // converts line breaks of VFM to <br> tags. default to 'false'.
    //   disableFormatHtml: true, // disables HTML formatting. default to 'false'.
  },
};

if (isPrint) {
  def.theme = [
    ...def.theme,
    // グレースケール印刷可能なPDF
    "./theme-nice-techbook/theme-print-pdf.css",
  ];
} else {
  def.theme = [
    ...def.theme,
    // オンライン向けのフルカラーPDF
    "./theme-nice-techbook/theme-online-pdf.css",
    "./theme-nice-techbook/theme-base/css/lib/prism/base.css",
    "./theme-nice-techbook/theme-base/css/lib/prism/theme-okaidia.css",
  ];
}

export default def;
