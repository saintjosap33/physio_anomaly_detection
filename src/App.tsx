import React from "react";
import { Switch, Route, Router as WouterRouter } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";

// Contexts
import { ThemeProvider } from "@/context/ThemeContext";
import { VitalsProvider } from "@/context/VitalsContext";

// Layout & Pages
import { Layout } from "@/components/Layout";
import { DashboardLive } from "@/pages/DashboardLive";
import { DetailedAnalytics } from "@/pages/DetailedAnalytics";
import { History } from "@/pages/History";
import { About } from "@/pages/About";
import NotFound from "@/pages/not-found";
import WatchAnomalyDetector from "@/pages/WatchAnomalyDetector";

const queryClient = new QueryClient();

function Router() {
  return (
    <Layout>
      <Switch>
        <Route path="/" component={WatchAnomalyDetector} />
        <Route path="/" component={DashboardLive} />
        <Route path="/analytics" component={DetailedAnalytics} />
        <Route path="/history" component={History} />
        <Route path="/about" component={About} />
        <Route component={NotFound} />
      </Switch>
    </Layout>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <VitalsProvider>
          <TooltipProvider>
            <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
              <Router />
            </WouterRouter>
            <Toaster />
          </TooltipProvider>
        </VitalsProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
