import { AnimatePresence, motion } from 'framer-motion';
import Sidebar from './components/Sidebar';
import ChatView from './components/ChatView';
import VoiceView from './components/VoiceView';
import SettingsView from './components/SettingsView';
import StatusBar from './components/StatusBar';
import { useAspire } from './hooks/useAspire';

export default function App() {
  const {
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
    sendMessage,
    clearChat,
  } = useAspire();

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ background: '#0d0d0f' }}
    >
      {/* Sidebar */}
      <Sidebar view={view} setView={setView} />

      {/* Main content */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* View area */}
        <div className="flex-1 overflow-hidden relative">
          <AnimatePresence mode="wait">
            {view === 'chat' && (
              <motion.div
                key="chat"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                className="absolute inset-0"
              >
                <ChatView
                  messages={messages}
                  aiState={aiState}
                  isWelcome={isWelcome}
                  drawer={drawer}
                  toggleDrawer={toggleDrawer}
                  inputValue={inputValue}
                  setInputValue={setInputValue}
                  onSend={sendMessage}
                  onClear={clearChat}
                  webEnabled={webEnabled}
                  setWebEnabled={setWebEnabled}
                  memoryEnabled={memoryEnabled}
                  setMemoryEnabled={setMemoryEnabled}
                  computerEnabled={computerEnabled}
                  setComputerEnabled={setComputerEnabled}
                  dragOver={dragOver}
                  setDragOver={setDragOver}
                />
              </motion.div>
            )}

            {view === 'voice' && (
              <motion.div
                key="voice"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                className="absolute inset-0"
              >
                <VoiceView />
              </motion.div>
            )}

            {view === 'settings' && (
              <motion.div
                key="settings"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                className="absolute inset-0"
              >
                <SettingsView />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Status bar */}
        <StatusBar
          aiState={aiState}
          memoryEnabled={memoryEnabled}
          webEnabled={webEnabled}
        />
      </div>
    </div>
  );
}
