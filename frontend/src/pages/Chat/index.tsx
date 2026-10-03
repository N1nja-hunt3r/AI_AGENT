import { useCallback, useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, Mic, ChevronDown, Plus, MessageSquare, Trash2, Search } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
import { MessageList } from "@/components/chat/MessageList";
import { ChatPipeline } from "@/components/chat/ChatPipeline";
import { VoiceMode } from "@/components/voice/VoiceMode";
import { streamMessage } from "@/api/chat";
import { useChatStore } from "@/store/chatStore";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/components/chat/types";
import type { PipelineStage } from "@/components/chat/ChatPipeline";

type Mode = "chat" | "voice";

const ChatPage: React.FC = () => {
  const [mode, setMode] = useState<Mode>("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>("planner");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const {
    conversations,
    currentConversationId,
    createConversation,
    selectConversation,
    deleteConversation,
    addMessage,
    appendToMessage,
    updateMessage,
  } = useChatStore();

  const [searchParams] = useSearchParams();

  useEffect(() => {
    const convId = searchParams.get("conv");
    if (convId && conversations[convId]) {
      selectConversation(convId);
      const conv = conversations[convId];
      if (conv) {
        const chatMessages: ChatMessage[] = conv.messages.map((m) => ({
          id: m.id,
          role: m.role as "user" | "assistant" | "system",
          content: typeof m.content === "string" ? m.content : "",
          status: m.status as "pending" | "streaming" | "complete" | "error",
          createdAt: new Date(m.createdAt).toISOString(),
          tokens: m.tokens?.total,
        }));
        setMessages(chatMessages);
      }
    }
  }, [searchParams]);

  const conversationList = Object.values(conversations)
    .filter((c) => (searchQuery ? c.title.toLowerCase().includes(searchQuery.toLowerCase()) : true))
    .sort((a, b) => b.updatedAt - a.updatedAt);

  const handleNewChat = useCallback(() => {
    createConversation({ title: "New Conversation" });
    setMessages([]);
  }, [createConversation]);

  const handleSelectConversation = useCallback(
    (id: string) => {
      selectConversation(id);
      const conv = conversations[id];
      if (conv) {
        setMessages(
          conv.messages.map((m) => ({
            id: m.id,
            role: m.role as "user" | "assistant" | "system",
            content: typeof m.content === "string" ? m.content : "",
            status: m.status as "pending" | "streaming" | "complete" | "error",
            createdAt: new Date(m.createdAt).toISOString(),
            tokens: m.tokens?.total,
          }))
        );
      }
      setSidebarOpen(false);
    },
    [selectConversation, conversations]
  );

  const simulatePipeline = useCallback(async () => {
    setPipelineStage("planner");
    await new Promise((r) => setTimeout(r, 600));
    setPipelineStage("classifier");
    await new Promise((r) => setTimeout(r, 500));
    setPipelineStage("agent");
    await new Promise((r) => setTimeout(r, 800));
    setPipelineStage("capability");
    await new Promise((r) => setTimeout(r, 600));
    setPipelineStage("response");
  }, []);

  const handleSendMessage = useCallback(
    async (content: string) => {
      let convId = currentConversationId;
      if (!convId) {
        convId = createConversation({ title: content.slice(0, 50) + (content.length > 50 ? "..." : "") });
      }

      const userMsg: ChatMessage = {
        id: `user_${Date.now()}`,
        role: "user",
        content,
        status: "complete",
        createdAt: new Date().toISOString(),
      };
      const assistantId = `asst_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "streaming",
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);
      simulatePipeline();

      addMessage(convId, { role: "user", content, status: "complete", tokens: undefined, metadata: undefined });
      addMessage(convId, { role: "assistant", content: "", status: "streaming", tokens: undefined, metadata: undefined });

      streamMessage({
        conversationId: convId,
        content,
        onChunk: (chunk) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + chunk.delta } : m))
          );
          appendToMessage(convId, assistantId, chunk.delta);
        },
        onComplete: () => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, status: "complete" } : m))
          );
          updateMessage(convId, assistantId, { status: "complete" });
          setIsStreaming(false);
        },
        onError: (error) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, status: "error", content: `Error: ${error.message}` } : m
            )
          );
          updateMessage(convId, assistantId, { status: "error" });
          setIsStreaming(false);
        },
      });
    },
    [currentConversationId, createConversation, addMessage, appendToMessage, updateMessage, simulatePipeline]
  );

  const handleStopStreaming = useCallback(() => {
    setIsStreaming(false);
  }, []);

  const handleVoiceTranscript = useCallback(
    (text: string) => {
      handleSendMessage(text);
    },
    [handleSendMessage]
  );

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-gray-950">
      {/* Mode Toggle */}
      <div className="flex items-center justify-center gap-1 px-4 pt-3 pb-2">
        <div className="flex rounded-lg border border-white/[0.06] bg-gray-900/60 p-0.5">
          <button
            onClick={() => setMode("chat")}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-all",
              mode === "chat"
                ? "bg-violet-600 text-white shadow-sm"
                : "text-gray-500 hover:text-gray-300"
            )}
          >
            <MessageCircle className="h-3.5 w-3.5" />
            Chat
          </button>
          <button
            onClick={() => setMode("voice")}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-all",
              mode === "voice"
                ? "bg-violet-600 text-white shadow-sm"
                : "text-gray-500 hover:text-gray-300"
            )}
          >
            <Mic className="h-3.5 w-3.5" />
            Voice
          </button>
        </div>

        <div className="ml-auto flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-gray-500 hover:text-gray-300 hover:bg-gray-800"
            onClick={handleNewChat}
          >
            <Plus className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-gray-500 hover:text-gray-300 hover:bg-gray-800"
            onClick={() => setSidebarOpen((p) => !p)}
          >
            <MessageSquare className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Pipeline visualization */}
      {isStreaming && (
        <div className="px-4 pb-2">
          <ChatPipeline activeStage={pipelineStage} className="rounded-lg border border-white/[0.04] bg-gray-900/40 px-3 py-2" />
        </div>
      )}

      {/* Conversation sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: 280 }}
            animate={{ x: 0 }}
            exit={{ x: 280 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute right-0 top-0 z-50 flex h-full w-72 flex-col border-l border-white/[0.06] bg-gray-950/95 backdrop-blur-2xl shadow-2xl"
          >
            <div className="flex items-center justify-between border-b border-white/[0.06] px-3 py-2.5">
              <span className="text-xs font-semibold text-gray-200">Conversations</span>
              <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-500" onClick={() => setSidebarOpen(false)}>
                <ChevronDown className="h-3 w-3" />
              </Button>
            </div>
            <div className="border-b border-white/[0.06] px-3 py-1.5">
              <div className="relative">
                <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-gray-600" />
                <Input
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-7 border-gray-800 bg-gray-900 pl-7 text-xs text-gray-300 placeholder:text-gray-600 focus:border-violet-500/50"
                />
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
              {conversationList.length === 0 ? (
                <p className="py-8 text-center text-xs text-gray-600">No conversations yet</p>
              ) : (
                conversationList.map((conv) => (
                  <div
                    key={conv.id}
                    onClick={() => handleSelectConversation(conv.id)}
                    className={cn(
                      "group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-xs transition-all",
                      conv.id === currentConversationId
                        ? "bg-violet-500/15 text-violet-300"
                        : "text-gray-500 hover:bg-gray-800/60 hover:text-gray-300"
                    )}
                  >
                    <MessageSquare className="h-3 w-3 shrink-0 text-gray-600" />
                    <span className="flex-1 truncate">{conv.title}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteConversation(conv.id);
                      }}
                      className="shrink-0 text-gray-700 opacity-0 group-hover:opacity-100 hover:text-red-400"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main chat / voice area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {mode === "chat" ? (
          <>
            <MessageList
              messages={messages}
              isTyping={isStreaming}
              onSendMessage={handleSendMessage}
              className="flex-1"
            />
            <div className="shrink-0 px-4 pb-3 pt-1">
              <ChatInput
                onSend={handleSendMessage}
                onStop={handleStopStreaming}
                isStreaming={isStreaming}
                placeholder="Ask Aspire anything…"
              />
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center">
            <VoiceMode
              onTranscript={handleVoiceTranscript}
              onStop={handleStopStreaming}
              isProcessing={isStreaming}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatPage;
