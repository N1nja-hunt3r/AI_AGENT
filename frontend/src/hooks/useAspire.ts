import { useState, useCallback, useRef, useEffect } from 'react';
import { eventManager } from '../services/event-manager';
import type { Message, AIState, AppView, DrawerType } from '../types';

export function useAspire() {
  const [view, setView] = useState<AppView>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [aiState, setAiState] = useState<AIState>('idle');
  const [drawer, setDrawer] = useState<DrawerType>(null);
  const [inputValue, setInputValue] = useState('');
  const [isWelcome, setIsWelcome] = useState(true);
  const [webEnabled, setWebEnabled] = useState(true);
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [computerEnabled, setComputerEnabled] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isStreaming) return;

    setIsWelcome(false);
    setIsStreaming(true);

    const userMsg: Message = {
      id: Math.random().toString(36).slice(2, 11),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');

    const assistantId = Math.random().toString(36).slice(2, 11);
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }]);

    let accumulated = '';

    await eventManager.streamMessage(content, { webEnabled, memoryEnabled, computerEnabled }, {
      onStateChange: (state) => {
        setAiState(state);
      },
      onChunk: (chunk) => {
        accumulated += (chunk.delta ?? '');
        setMessages(prev =>
          prev.map(m => m.id === assistantId ? { ...m, content: accumulated } : m),
        );
      },
      onError: (error) => {
        console.error('EventManager error:', error);
        accumulated = accumulated || error.message;
        setMessages(prev =>
          prev.map(m => m.id === assistantId ? { ...m, content: accumulated } : m),
        );
      },
    });

    setIsStreaming(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setAiState('idle'), 2000);
  }, [webEnabled, memoryEnabled, computerEnabled, isStreaming]);

  const clearChat = useCallback(() => {
    eventManager.clearConversation();
    setMessages([]);
    setIsWelcome(true);
    setAiState('idle');
    setIsStreaming(false);
  }, []);

  const toggleDrawer = useCallback((type: DrawerType) => {
    setDrawer(prev => prev === type ? null : type);
  }, []);

  return {
    view, setView,
    messages,
    aiState,
    drawer, toggleDrawer,
    inputValue, setInputValue,
    isWelcome,
    webEnabled, setWebEnabled,
    memoryEnabled, setMemoryEnabled,
    computerEnabled, setComputerEnabled,
    dragOver, setDragOver,
    isStreaming,
    sendMessage,
    clearChat,
  };
}
