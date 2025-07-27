// deno run --allow-read .github/scripts/CountFiles.ts
// vivliostyle.config.jsから章ファイル数をカウントする

/**
 * vivliostyle.config.jsを読み込んでchapterファイル数を取得
 */
async function countChapterFiles(): Promise<number> {
  try {
    // vivliostyle.config.jsを動的にimport
    const configPath = new URL('../../vivliostyle.config.js', import.meta.url);
    const config = await import(configPath.href);
    
    // entry配列からchap-で始まる文字列ファイルのみを抽出
    const entries = config.default.entry || [];
    const chapterFiles = entries.filter((entry: string | object) => {
      return typeof entry === 'string' && entry.startsWith('chap-') && entry.endsWith('.md');
    });
    
    console.log(`Found chapter files: ${chapterFiles.join(', ')}`);
    return chapterFiles.length;
    
  } catch (error) {
    console.error('Error reading vivliostyle.config.js:', error);
    throw error;
  }
}

// CLI実行
try {
  const chapterCount = await countChapterFiles();
  console.log(`Chapter files count: ${chapterCount}`);
} catch (error) {
  console.error('Failed to count files:', error);
  Deno.exit(1);
}