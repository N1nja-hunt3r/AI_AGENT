import { useCallback } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Settings, Sparkles, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/store/chatStore";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

interface NavItem {
  label: string;
  icon: React.ElementType;
  path: string;
}

const NAV_ITEMS: NavItem[] = [
  { label: "Chat", icon: MessageSquare, path: "/chat" },
];

const BOTTOM_ITEMS: NavItem[] = [
  { label: "Settings", icon: Settings, path: "/settings" },
];

const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggle }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const conversations = useChatStore((s) => s.conversations);

  const isActive = useCallback((path: string) => location.pathname.startsWith(path), [location.pathname]);

  const recentConversations = Object.values(conversations)
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 5);

  return (
    <div
      className={cn(
        "relative flex h-full flex-col border-r border-white/[0.06] bg-gray-900/80 backdrop-blur-xl transition-all duration-300 ease-out"
      )}
      style={{ width: collapsed ? 64 : 260 }}
      role="navigation"
    >
      <div className="flex h-14 shrink-0 items-center px-4">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-linear-to-br from-violet-500 to-indigo-600 shadow-lg shadow-violet-500/25 ring-1 ring-white/10">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <AnimatePresence initial={false}>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
              >
                <span className="text-sm font-semibold tracking-tight text-white">Nexus AI</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <button
        onClick={onToggle}
        className={cn(
          "absolute right-0 top-[3.75rem] z-10 translate-x-1/2 flex h-5 w-5 items-center justify-center rounded-full border border-white/10 bg-gray-800 text-gray-400 transition-colors hover:bg-gray-700 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500"
        )}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
      </button>

      <div className="mx-3 h-px bg-white/[0.06]" />

      <nav className="flex flex-col gap-0.5 overflow-y-auto overflow-x-hidden px-2 py-2">
        {NAV_ITEMS.map(({ label, icon: Icon, path }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive: active }) =>
              cn(
                "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500",
                active || isActive(path)
                  ? "bg-violet-500/15 text-violet-300"
                  : "text-gray-400 hover:bg-white/[0.05] hover:text-gray-200"
              )
            }
            title={collapsed ? label : undefined}
          >
            {({ isActive: active }) => (
              <>
                {(active || isActive(path)) && (
                  <motion.div
                    layoutId="active-nav-indicator"
                    className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-violet-400"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon className={cn("h-4 w-4 shrink-0", active || isActive(path) ? "text-violet-400" : "text-gray-500")} />
                <AnimatePresence initial={false}>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                      transition={{ duration: 0.2 }}
                      className="flex-1 truncate text-sm font-medium"
                    >
                      {label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mx-3 h-px bg-white/[0.06]" />

      <AnimatePresence initial={false}>
        {!collapsed && recentConversations.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="flex items-center gap-2 px-4 pt-2 pb-1">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-500">Recent</span>
            </div>
            <div className="flex flex-col gap-0.5 px-2 pb-2">
              {recentConversations.map((conv) => (
                <button
                  key={conv.id}
                  onClick={() => navigate(`/chat?conv=${conv.id}`)}
                  className="group flex items-center gap-2 rounded-lg px-3 py-1.5 text-left text-gray-500 transition-all hover:bg-white/[0.04] hover:text-gray-300"
                  title={conv.title}
                >
                  <MessageSquare className="h-3 w-3 shrink-0 text-gray-600" />
                  <span className="flex-1 truncate text-xs">{conv.title}</span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-auto flex flex-col gap-0.5 px-2 py-3">
        {BOTTOM_ITEMS.map(({ label, icon: Icon, path }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive: active }) =>
              cn(
                "group flex items-center gap-3 rounded-lg px-3 py-2.5 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500",
                active ? "bg-violet-500/15 text-violet-300" : "text-gray-400 hover:bg-white/[0.05] hover:text-gray-200"
              )
            }
            title={collapsed ? label : undefined}
          >
            <Icon className="h-4 w-4 shrink-0 text-gray-500" />
            <AnimatePresence initial={false}>
              {!collapsed && (
                <motion.span
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.2 }}
                  className="text-sm font-medium"
                >
                  {label}
                </motion.span>
              )}
            </AnimatePresence>
          </NavLink>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
