import { ChatService } from './chat.service';
import type { StreamChunk, TokenUsage } from './chat.service';
import type { AIState, Message } from '../types';

export interface EventContext {
  webEnabled: boolean;
  memoryEnabled: boolean;
  computerEnabled: boolean;
  conversationId?: string;
  userId?: string;
}

export interface EventResult {
  message: Message;
  conversationId: string;
  usage?: TokenUsage;
}

export type EventCallback = {
  onStateChange: (state: AIState) => void;
  onChunk?: (chunk: StreamChunk) => void;
  onError?: (error: Error) => void;
};

function validateInput(input: string): string | null {
  const trimmed = input.trim();
  if (!trimmed) return 'Message cannot be empty';
  if (trimmed.length > 1_000_000) return 'Message too long (max 1,000,000 characters)';
  return null;
}

function classifyIntent(input: string): AIState {
  const lower = input.toLowerCase();
  if (lower.includes('code') || lower.includes('function') || lower.includes('implement') || lower.includes('write a ')) return 'coding';
  if (lower.includes('search') || lower.includes('find') || lower.includes('look up') || lower.includes('google')) return 'searching';
  if (lower.includes('remember') || lower.includes('save') || lower.includes('store') || lower.includes('memory')) return 'remembering';
  if (lower.includes('read') || lower.includes('file') || lower.includes('document') || lower.includes('open')) return 'reading';
  if (lower.includes('browse') || lower.includes('navigate') || lower.includes('go to') || lower.includes('website')) return 'browsing';
  return 'thinking';
}

function buildSystemPrompt(context: EventContext): string {
  const parts: string[] = ['You are Aspire, a personal AI operating system assistant.'];
  if (context.webEnabled) parts.push('You can search the web for current information when needed.');
  if (context.memoryEnabled) parts.push('You have access to conversation memory to remember context.');
  if (context.computerEnabled) parts.push('You can interact with the computer (screen, mouse, keyboard, terminal).');
  parts.push('Keep responses concise, accurate, and helpful.');
  return parts.join(' ');
}

export class FrontendEventManager {
  private abortController: AbortController | null = null;
  private conversationId: string | null = null;

  get currentConversationId(): string | null {
    return this.conversationId;
  }

  cancel(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  async sendMessage(
    content: string,
    context: EventContext,
    callbacks: EventCallback,
  ): Promise<EventResult | null> {
    const validationError = validateInput(content);
    if (validationError) {
      callbacks.onStateChange('idle');
      if (callbacks.onError) {
        callbacks.onError(new Error(validationError));
      }
      return null;
    }

    const initialState = classifyIntent(content);
    callbacks.onStateChange(initialState);

    try {
      const systemPrompt = buildSystemPrompt(context);
      const metadata: Record<string, unknown> = {
        web_enabled: context.webEnabled,
        memory_enabled: context.memoryEnabled,
        computer_enabled: context.computerEnabled,
      };

      const result = await ChatService.sendMessage({
        conversationId: context.conversationId ?? this.conversationId ?? undefined,
        content: content.trim(),
        systemPrompt,
        metadata,
      });

      this.conversationId = result.conversationId;

      const assistantMsg: Message = {
        id: result.message.id,
        role: 'assistant',
        content: result.message.content,
        timestamp: new Date(result.message.createdAt),
        status: 'finished' as const,
      };

      callbacks.onStateChange('finished');
      setTimeout(() => callbacks.onStateChange('idle'), 2000);

      return {
        message: assistantMsg,
        conversationId: result.conversationId,
        usage: result.usage,
      };
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Failed to send message');
      callbacks.onStateChange('finished');
      if (callbacks.onError) callbacks.onError(err);
      setTimeout(() => callbacks.onStateChange('idle'), 1000);
      return null;
    }
  }

  streamMessage(
    content: string,
    context: EventContext,
    callbacks: EventCallback & {
      onChunk: (chunk: StreamChunk) => void;
    },
  ): Promise<void> {
    const validationError = validateInput(content);
    if (validationError) {
      callbacks.onStateChange('idle');
      if (callbacks.onError) callbacks.onError(new Error(validationError));
      return Promise.resolve();
    }

    const initialState = classifyIntent(content);
    callbacks.onStateChange(initialState);

    const systemPrompt = buildSystemPrompt(context);
    const metadata: Record<string, unknown> = {
      web_enabled: context.webEnabled,
      memory_enabled: context.memoryEnabled,
      computer_enabled: context.computerEnabled,
    };

    return new Promise<void>((resolve) => {
      this.abortController = ChatService.streamMessage({
        conversationId: context.conversationId ?? this.conversationId ?? undefined,
        content: content.trim(),
        systemPrompt,
        metadata,
        onChunk: (chunk: StreamChunk) => {
          callbacks.onChunk(chunk);
        },
        onComplete: () => {
          this.abortController = null;
          callbacks.onStateChange('finished');
          setTimeout(() => callbacks.onStateChange('idle'), 2000);
          resolve();
        },
        onError: (error: Error) => {
          this.abortController = null;
          callbacks.onStateChange('finished');
          if (callbacks.onError) callbacks.onError(error);
          setTimeout(() => callbacks.onStateChange('idle'), 1000);
          resolve();
        },
      });
    });
  }

  clearConversation(): void {
    this.conversationId = null;
    this.cancel();
  }
}

export const eventManager = new FrontendEventManager();
