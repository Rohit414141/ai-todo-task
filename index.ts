import "dotenv/config";
import readline from "node:readline/promises";
import Anthropic from "@anthropic-ai/sdk";
import { eq, ilike, desc } from "drizzle-orm";
import { db } from "./db/index.ts";
import { todosTable } from "./db/schema.ts";

// ---------------------------------------------------------------------------
// Tools (the functions the agent can call)
// ---------------------------------------------------------------------------

async function getAllTodos() {
  return db.select().from(todosTable).orderBy(desc(todosTable.id));
}

async function createTodo(todo: string) {
  const [row] = await db
    .insert(todosTable)
    .values({ todo })
    .returning({ id: todosTable.id });
  return row.id;
}

async function searchTodo(search: string) {
  return db
    .select()
    .from(todosTable)
    .where(ilike(todosTable.todo, `%${search}%`));
}

async function deleteTodoById(id: number) {
  const deleted = await db
    .delete(todosTable)
    .where(eq(todosTable.id, id))
    .returning({ id: todosTable.id });
  return deleted.length > 0;
}

// Tool definitions sent to Claude. Claude decides which to call (ACTION),
// we run it and send the result back (OBSERVATION).
const tools: Anthropic.Beta.BetaTool[] = [
  {
    name: "getAllTodos",
    description: "Returns all the todos from the database.",
    input_schema: { type: "object", properties: {}, additionalProperties: false },
    strict: true,
  },
  {
    name: "createTodo",
    description:
      "Creates a new todo in the database and returns the ID of the created todo.",
    input_schema: {
      type: "object",
      properties: { todo: { type: "string", description: "The todo text" } },
      required: ["todo"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    name: "searchTodo",
    description:
      "Searches todos whose text contains the given query (case-insensitive) and returns the matches.",
    input_schema: {
      type: "object",
      properties: { search: { type: "string", description: "Text to search for" } },
      required: ["search"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    name: "deleteTodoById",
    description: "Deletes a todo by its ID. Returns whether a todo was deleted.",
    input_schema: {
      type: "object",
      properties: { id: { type: "integer", description: "The todo ID" } },
      required: ["id"],
      additionalProperties: false,
    },
    strict: true,
  },
];

async function runTool(name: string, input: any): Promise<unknown> {
  switch (name) {
    case "getAllTodos":
      return getAllTodos();
    case "createTodo":
      return { id: await createTodo(String(input.todo)) };
    case "searchTodo":
      return searchTodo(String(input.search));
    case "deleteTodoById":
      return { deleted: await deleteTodoById(Number(input.id)) };
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// ---------------------------------------------------------------------------
// System prompt (adapted from the START / PLAN / ACTION / OBSERVATION prompt)
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are an AI To-Do List Assistant that works in START, PLAN, ACTION, OBSERVATION and OUTPUT steps.
Wait for the user prompt (START) and first PLAN using the available tools.
After planning, take the ACTION by calling the appropriate tool and wait for the OBSERVATION (the tool result).
Once you have the observations, return the OUTPUT: a short, friendly answer based on the user's prompt and the observations.

You can manage tasks by adding, viewing, searching and deleting them.
Before deleting by description, search first to find the correct ID. If several todos match, ask the user which one.

Todo DB Schema:
id: Int and Primary Key
todo: String
created_at: Date Time
updated_at: Date Time

Available Tools:
- getAllTodos(): Returns all the todos from the database
- createTodo(todo: string): Creates a new todo in the DB and returns the ID of the created todo
- searchTodo(search: string): Searches for all todos matching the query string
- deleteTodoById(id: number): Deletes the todo by ID`;

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

const client = new Anthropic(); // reads ANTHROPIC_API_KEY from the environment / .env
const messages: Anthropic.Beta.BetaMessageParam[] = [];

async function handleUserPrompt(userPrompt: string) {
  // START
  messages.push({ role: "user", content: userPrompt });

  while (true) {
    const response = await client.beta.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 16000,
      output_config: { effort: "low" }, // simple CRUD tasks don't need deep thinking
      // If a safety classifier declines, the API retries on a fallback model.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system: SYSTEM_PROMPT,
      tools,
      messages,
    });

    // Keep the full assistant content (incl. thinking/tool_use blocks) in history.
    messages.push({ role: "assistant", content: response.content });

    if (response.stop_reason === "refusal") {
      console.log("🤖: Sorry, I can't help with that request.");
      return;
    }

    const toolCalls = response.content.filter(
      (b): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use",
    );

    // OUTPUT: no more tool calls -> print the final answer.
    if (toolCalls.length === 0) {
      const text = response.content
        .filter((b): b is Anthropic.Beta.BetaTextBlock => b.type === "text")
        .map((b) => b.text)
        .join("\n");
      console.log(`🤖: ${text}`);
      return;
    }

    // ACTION -> OBSERVATION: run every requested tool, return all results in one message.
    const toolResults: Anthropic.Beta.BetaToolResultBlockParam[] = [];
    for (const call of toolCalls) {
      console.log(`  ⚙️  ACTION: ${call.name}(${JSON.stringify(call.input)})`);
      try {
        const result = await runTool(call.name, call.input);
        console.log(`  👀 OBSERVATION: ${JSON.stringify(result)}`);
        toolResults.push({
          type: "tool_result",
          tool_use_id: call.id,
          content: JSON.stringify(result),
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        console.log(`  ❌ ERROR: ${message}`);
        toolResults.push({
          type: "tool_result",
          tool_use_id: call.id,
          content: message,
          is_error: true,
        });
      }
    }
    messages.push({ role: "user", content: toolResults });
  }
}

async function main() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  rl.setPrompt(">> ");
  let inputClosed = false;
  rl.on("close", () => (inputClosed = true));
  const prompt = () => {
    if (!inputClosed) rl.prompt();
  };
  console.log('AI Todo Agent ready. Type a request (or "exit" to quit).');
  prompt();

  for await (const line of rl) {
    const userPrompt = line.trim();
    if (userPrompt === "exit") break;
    if (!userPrompt) {
      prompt();
      continue;
    }
    console.log(`\n🧑: ${userPrompt}`);

    try {
      await handleUserPrompt(userPrompt);
    } catch (err) {
      if (err instanceof Anthropic.AuthenticationError) {
        console.error("Invalid ANTHROPIC_API_KEY - check your .env file.");
      } else if (err instanceof Anthropic.RateLimitError) {
        console.error("Rate limited - try again in a moment.");
      } else if (err instanceof Anthropic.APIError) {
        console.error(`API error ${err.status}: ${err.message}`);
      } else {
        console.error(err);
      }
    }
    prompt();
  }

  rl.close();
  process.exit(0);
}

main();
