// deno run --allow-env --allow-net .github/scripts/notify-stats.ts 10 15
// 引数は変更前のチャプター数と変更後のチャプター数

function buildMessage(prevCount: number, currentCount: number) {
  const diff = currentCount - prevCount
  const diffMessage = diff > 0 ? `（前日比 +${diff}）` : ''
  return {
    "blocks": [
      {
        "type": "context",
        "elements": [
          {
            "type": "mrkdwn",
            "text": `『<https://github.com/onestop-techbook/Onestop-gourmet|ワンストップ食>』の執筆状況: 現在 *${currentCount}* 章 ${diffMessage} :books: <https://oyakata-techbook.slack.com/canvas/CDS3UGG75|執筆の手引き>`
          }
        ]
      },
      {
        "type": "context",
        "elements": [
          {
            "type": "plain_text",
            "text": `${':page_with_curl:'.repeat(currentCount - diff)}${':new:'.repeat(diff)}`,
            "emoji": true
          }
        ]
      }
    ]
  }
}

// Slack通知を送信する関数
async function sendSlackNotification(webhookUrl: string, message: any) {
  const response = await fetch(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(message),
  });

  if (!response.ok) {
    console.error("Failed to send Slack notification:", await response.text());
    Deno.exit(1);
  }
}

async function main(prevCount: number, currentCount: number) {
  const webhookUrl = Deno.env.get("SLACK_WEBHOOK_URL");
  if (!webhookUrl) {
    console.error("SLACK_WEBHOOK_URL is not set.");
    Deno.exit(1);
  }

  if (isNaN(prevCount) || isNaN(currentCount)) {
    console.error("Usage: deno run --allow-read --allow-write .github/scripts/stats.ts <prevCount> <currentCount>");
    Deno.exit(1);
  }

  if (currentCount - prevCount < 0) {
    console.error("チャプター数が減少しているため、処理を終了します。");
    Deno.exit(1);
  }

  if (currentCount === prevCount) {
    console.log("チャプター数に変化がないので、処理を終了します。");
    Deno.exit(0);
  }

  const message = buildMessage(prevCount, currentCount);
  await sendSlackNotification(webhookUrl, message);
}

const [prevCount, currentCount] = Deno.args.map(Number);
main(prevCount, currentCount)