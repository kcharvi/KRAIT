"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User } from "lucide-react";
import MarkdownRenderer from "./MarkdownRenderer";

interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
}

export default function ChatNew() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: input.trim(),
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);

        // Create streaming message
        const streamingMessageId = (Date.now() + 1).toString();
        const streamingMessage: Message = {
            id: streamingMessageId,
            role: "assistant",
            content: "",
            timestamp: new Date(),
            isStreaming: true,
        };

        setMessages((prev) => [...prev, streamingMessage]);

        try {
            // Get the full response first
            const response = await fetch(`${backendUrl}/api/v1/chat/completions`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    messages: [...messages, userMessage].map((msg) => ({
                        role: msg.role,
                        content: msg.content,
                    })),
                    model: "gemini-1.5-flash",
                    provider: "gemini",
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to send message");
            }

            const data = await response.json();
            const fullContent = data.content || "Sorry, I could not process your message.";

            // Simulate streaming by updating the message progressively
            const words = fullContent.split(" ");
            let currentContent = "";

            for (let i = 0; i < words.length; i++) {
                currentContent += (i > 0 ? " " : "") + words[i];
                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === streamingMessageId ? { ...msg, content: currentContent } : msg
                    )
                );
                await new Promise((resolve) => setTimeout(resolve, 100)); // 100ms delay between words
            }

            // Mark as done
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === streamingMessageId ? { ...msg, isStreaming: false } : msg
                )
            );
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === streamingMessageId
                        ? {
                              ...msg,
                              content:
                                  "Sorry, there was an error processing your message. Please try again.",
                              isStreaming: false,
                          }
                        : msg
                )
            );
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-white">
            {/* Header */}
            <div className="flex items-center px-6 py-4 border-b border-gray-200">
                <Bot className="h-6 w-6 text-primary-500 mr-3" />
                <h2 className="text-lg font-semibold text-gray-900">Chat Assistant</h2>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                            <Bot className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                            <p className="text-gray-500">
                                Start a conversation with the AI assistant
                            </p>
                        </div>
                    </div>
                ) : (
                    messages.map((message) => (
                        <div
                            key={message.id}
                            className={`flex ${
                                message.role === "user" ? "justify-end" : "justify-start"
                            }`}
                        >
                            <div className={`flex items-start space-x-2 max-w-xs lg:max-w-2xl`}>
                                {message.role === "assistant" && (
                                    <div className="flex-shrink-0">
                                        <Bot className="h-6 w-6 text-primary-500" />
                                    </div>
                                )}
                                <div
                                    className={`chat-message ${
                                        message.role === "user" ? "user" : "assistant"
                                    }`}
                                >
                                    {message.role === "assistant" ? (
                                        <div className="text-sm">
                                            <MarkdownRenderer content={message.content} />
                                            {message.isStreaming && (
                                                <span className="inline-block w-2 h-4 bg-primary-500 animate-pulse ml-1" />
                                            )}
                                        </div>
                                    ) : (
                                        <p className="text-sm">{message.content}</p>
                                    )}
                                    <p className="text-xs opacity-70 mt-1">
                                        {message.timestamp.toLocaleTimeString()}
                                    </p>
                                </div>
                                {message.role === "user" && (
                                    <div className="flex-shrink-0">
                                        <User className="h-6 w-6 text-primary-500" />
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
                {isLoading && (
                    <div className="flex justify-start">
                        <div className="flex items-start space-x-2">
                            <Bot className="h-6 w-6 text-primary-500" />
                            <div className="chat-message assistant">
                                <div className="flex space-x-1">
                                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                    <div
                                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                                        style={{ animationDelay: "0.1s" }}
                                    ></div>
                                    <div
                                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                                        style={{ animationDelay: "0.2s" }}
                                    ></div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-gray-200 p-4">
                <form onSubmit={sendMessage} className="flex space-x-4">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type your message..."
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        <Send className="h-5 w-5" />
                    </button>
                </form>
            </div>
        </div>
    );
}
