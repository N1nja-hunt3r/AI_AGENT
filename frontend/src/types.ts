export type AppView = 'chat' | 'voice' | 'settings';

export type AIState =
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'searching'
  | 'coding'
  | 'reading'
  | 'browsing'
  | 'remembering'
  | 'speaking'
  | 'finished';

export type DrawerType =
  | null
  | 'memory'
  | 'files'
  | 'tasks'
  | 'reasoning'
  | 'tokens';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  status?: AIState;
  attachments?: Attachment[];
}

export interface Attachment {
  id: string;
  name: string;
  type: 'image' | 'file' | 'code';
  size?: string;
  preview?: string;
}

export interface CapabilityCard {
  id: string;
  icon: string;
  label: string;
  prompt: string;
  color: string;
}

export interface StatusInfo {
  online: boolean;
  provider: string;
  model: string;
  latency: number;
  voiceActive: boolean;
  memoryActive: boolean;
}
