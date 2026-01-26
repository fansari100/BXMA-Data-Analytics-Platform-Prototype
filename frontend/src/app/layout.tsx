import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
  title: "BXMA Risk/Quant Platform",
  description: "Blackstone Multi-Asset Investing Risk & Quantitative Analytics Platform",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased bg-dark-950 text-dark-100">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
