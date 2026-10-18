import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "two-tier gateway",
  description: "live telemetry and ambient feedback loop",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
