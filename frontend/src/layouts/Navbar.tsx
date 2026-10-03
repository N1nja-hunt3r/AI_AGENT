import { useLocation } from "react-router-dom";
import { Menu } from "lucide-react";
import { cn } from "@/lib/utils";

interface NavbarProps {
  onMenuToggle: () => void;
  sidebarCollapsed: boolean;
}

const PAGE_TITLES: Record<string, string> = {
  "/chat": "Chat",
  "/settings": "Settings",
};

const Navbar: React.FC<NavbarProps> = ({ onMenuToggle, sidebarCollapsed }) => {
  const location = useLocation();
  const currentPage = PAGE_TITLES[location.pathname] || "Nexus AI";

  return (
    <div
      className={cn(
        "flex h-12 w-full items-center gap-3 px-4 md:px-6 border-b border-white/[0.06] bg-gray-900/70 backdrop-blur-xl"
      )}
      role="banner"
    >
      <button
        onClick={onMenuToggle}
        className={cn(
          "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-white/[0.06] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500"
        )}
        aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <Menu className="h-4 w-4" />
      </button>

      <h1 className="truncate text-sm font-semibold text-white tracking-tight">{currentPage}</h1>
    </div>
  );
};

export default Navbar;
