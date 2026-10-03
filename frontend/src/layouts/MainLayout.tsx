import { useState, useCallback, useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";

const SIDEBAR_EXPANDED = 260;
const SIDEBAR_COLLAPSED = 64;
const BREAKPOINT = 1024;

const MainLayout: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < BREAKPOINT);
  const [mobileOpen, setMobileOpen] = useState(false);

  const toggleSidebar = useCallback(() => {
    if (isMobile) setMobileOpen((p) => !p);
    else setSidebarCollapsed((p) => !p);
  }, [isMobile]);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < BREAKPOINT;
      setIsMobile(mobile);
      if (mobile) setMobileOpen(false);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const sidebarWidth = sidebarCollapsed ? SIDEBAR_COLLAPSED : SIDEBAR_EXPANDED;
  const location = useLocation();
  const isChat = location.pathname.startsWith("/chat");

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-950 text-gray-100 antialiased">
      <AnimatePresence>
        {isMobile && mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
        )}
      </AnimatePresence>

      {isMobile ? (
        <AnimatePresence>
          {mobileOpen && (
            <motion.aside
              initial={{ x: -SIDEBAR_EXPANDED }}
              animate={{ x: 0 }}
              exit={{ x: -SIDEBAR_EXPANDED }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
              className="fixed left-0 top-0 z-50 h-full"
            >
              <Sidebar collapsed={false} onToggle={() => setMobileOpen(false)} />
            </motion.aside>
          )}
        </AnimatePresence>
      ) : (
        <motion.aside
          animate={{ width: sidebarWidth }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="relative z-30 h-full shrink-0 overflow-hidden"
        >
          <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} />
        </motion.aside>
      )}

      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="z-20 shrink-0">
          <Navbar onMenuToggle={toggleSidebar} sidebarCollapsed={sidebarCollapsed} />
        </header>

        <main className={cn("relative flex flex-1 overflow-hidden", isChat && "pt-0")} id="main-content">
          <motion.div
            className={cn("flex flex-1 flex-col", isChat ? "overflow-hidden" : "overflow-y-auto")}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          >
            <div className={cn("flex-1", isChat ? "flex flex-col overflow-hidden" : "px-4 py-6 md:px-6 lg:px-8")}>
              <Outlet />
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
