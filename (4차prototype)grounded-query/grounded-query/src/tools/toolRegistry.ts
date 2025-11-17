import { ToolConfig, ToolResult } from '@/types';

/**
 * Tool Registry
 * 
 * Central registry for all external tools (web search, calculator, etc.)
 * Provides abstraction for tool calling with fallback support.
 * 
 * Integration points for MCP:
 * - MCP servers can register tools here
 * - Tool discovery from MCP endpoints
 * - Unified interface for tool calling regardless of source
 * 
 * TODO: Implement MCP handshake and dynamic tool loading
 * See: tools/mcpRegistry.ts for MCP-specific integration
 */

export interface Tool {
  name: string;
  description: string;
  execute(params: Record<string, unknown>): Promise<ToolResult>;
  config: ToolConfig;
}

// Mock tool implementations
const webSearchTool: Tool = {
  name: 'web_search',
  description: 'Search the web for current information',
  config: { enabled: true, name: 'web_search', description: 'Web search' },
  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const query = typeof params.query === 'string' ? params.query : '';
    await new Promise(resolve => setTimeout(resolve, 600));
    return {
      toolName: 'web_search',
      query,
      type: 'web_search',
      timestamp: new Date(),
      result: {
        results: [
          {
            title: 'Example Search Result 1',
            snippet: `Information about "${query}" from the web...`,
            url: 'https://example.com/result1',
          },
          {
            title: 'Example Search Result 2',
            snippet: `More details about "${query}"...`,
            url: 'https://example.com/result2',
          },
        ],
      },
    };
  },
};

const calculatorTool: Tool = {
  name: 'calculator',
  description: 'Perform mathematical calculations',
  config: { enabled: true, name: 'calculator', description: 'Calculator' },
  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const expression = typeof params.expression === 'string' ? params.expression : '';
    await new Promise(resolve => setTimeout(resolve, 200));
    try {
      // Simple mock calculation
      const result = Function(`"use strict"; return (${expression})`)();
      return {
        toolName: 'calculator',
        query: expression,
        type: 'calculator',
        timestamp: new Date(),
        result: { value: result },
      };
    } catch (error) {
      return {
        toolName: 'calculator',
        query: expression,
        type: 'calculator',
        timestamp: new Date(),
        result: { error: 'Invalid expression' },
      };
    }
  },
};

const fetchUrlTool: Tool = {
  name: 'fetch_url',
  description: 'Fetch content from a URL',
  config: { enabled: true, name: 'fetch_url', description: 'Fetch URL' },
  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    const url = typeof params.url === 'string' ? params.url : '';
    await new Promise(resolve => setTimeout(resolve, 800));
    return {
      toolName: 'fetch_url',
      query: url,
      type: 'fetch_url',
      timestamp: new Date(),
      result: {
        content: `Mock content from ${url}`,
        title: 'Example Page',
      },
    };
  },
};

class ToolRegistry {
  private tools: Map<string, Tool> = new Map();

  constructor() {
    // Register default tools
    this.register(webSearchTool);
    this.register(calculatorTool);
    this.register(fetchUrlTool);
  }

  register(tool: Tool) {
    this.tools.set(tool.name, tool);
  }

  async executeTool(toolName: string, params: Record<string, unknown>): Promise<ToolResult> {
    const tool = this.tools.get(toolName);
    if (!tool) {
      throw new Error(`Tool ${toolName} not found`);
    }
    if (!tool.config.enabled) {
      throw new Error(`Tool ${toolName} is disabled`);
    }
    return tool.execute(params);
  }

  getTools(): Tool[] {
    return Array.from(this.tools.values());
  }

  getEnabledTools(): Tool[] {
    return this.getTools().filter(t => t.config.enabled);
  }

  updateToolConfig(toolName: string, config: Partial<ToolConfig>) {
    const tool = this.tools.get(toolName);
    if (tool) {
      tool.config = { ...tool.config, ...config };
    }
  }
}

export const toolRegistry = new ToolRegistry();
