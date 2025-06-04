import React, { useState, useEffect } from 'react';
import {
  ChatBubbleLeftIcon,
  PaperAirplaneIcon,
  Cog6ToothIcon,
  PlusCircleIcon,
  TrashIcon,
  PencilIcon,
  CheckIcon,
  XMarkIcon,
  ArrowPathIcon,
  DocumentArrowDownIcon,
} from '@heroicons/react/24/outline';
import ReactMarkdown from 'react-markdown';
import { models } from './models';
import { PDFDownloadLink, Document, Page, Text, View, StyleSheet } from '@react-pdf/renderer';

const styles = StyleSheet.create({
  page: {
    padding: 30,
  },
  message: {
    marginBottom: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  content: {
    fontSize: 12,
    lineHeight: 1.5,
  },
  timestamp: {
    fontSize: 10,
    color: '#666',
    marginTop: 5,
  },
});

const ChatPDF = ({ messages, selectedModel }) => (
  <Document>
    <Page size="A4" style={styles.page}>
      {messages.map((message) => (
        <View key={message.id} style={styles.message}>
          <View style={styles.header}>
            <Text>{message.role === 'user' ? 'You' : selectedModel.name}</Text>
            <Text>{new Date(message.timestamp).toLocaleString()}</Text>
          </View>
          <Text style={styles.content}>{message.content}</Text>
        </View>
      ))}
    </Page>
  </Document>
);

export const themes = [
  { id: 'dark', name: 'Dark Theme' },
  { id: 'light', name: 'Light Theme' },
  { id: 'system', name: 'System Default' },
];

export const fontSizes = [
  { id: 'sm', name: 'Small' },
  { id: 'md', name: 'Medium' },
  { id: 'lg', name: 'Large' },
];

function App() {
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [searchLength, setSearchLength] = useState(20);
  const [extensionSize, setExtensionSize] = useState(100);
  const [tokenError, setTokenError] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStep, setGenerationStep] = useState('');
  const [chats, setChats] = useState([
    { id: 1, name: 'New Chat 1', messages: [] },
    { id: 2, name: 'New Chat 2', messages: [] },
  ]);
  const [selectedChat, setSelectedChat] = useState(chats[0]);
  const [showSidebar, setShowSidebar] = useState(true);
  const [activeTab, setActiveTab] = useState('chats');
  const [theme, setTheme] = useState(themes[0]);
  const [fontSize, setFontSize] = useState(fontSizes[1]);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingChatName, setEditingChatName] = useState('');

  const isDarkTheme = theme.id.startsWith('dark');
  const settingsImage = isDarkTheme
    ? '/CROssBAR_logo_dark_transparent.svg'
    : '/CROssBAR_logo_light_transparent.svg';

  useEffect(() => {
    document.documentElement.className = theme.id;
  }, [theme]);

  const simulateResponseGeneration = async () => {
    setIsGenerating(true);
    
    const steps = [
      'Analyzing your question...',
      'Searching through nodes...',
      'Generating graphs...',
      'Comparing graphs...',
      'Generating response...'
    ];

    try {
      for (const step of steps) {
        setGenerationStep(step);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      return true;
    } catch (error) {
      setTokenError(error.message);
      return true;
    } finally {
      setIsGenerating(false);
      setGenerationStep('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages([...messages, newMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: selectedChat.id,
          content: input,
          model: selectedModel.id,
          api_key: apiKey,
          searchLength: searchLength,
          extensionSize: extensionSize,
          //retrieved_docs: [], // TODO: Replace with real docs once retrieval is wired
        }),
      });

      const data = await response.json();

      const aiResponse = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
      };

      const updatedMessages = [...messages, newMessage, aiResponse];
      setMessages(updatedMessages);

      const updatedChats = chats.map((chat) =>
        chat.id === selectedChat.id
          ? { ...chat, messages: updatedMessages }
          : chat
      );
      setChats(updatedChats);
    } catch (error) {
      setTokenError('Failed to connect to the backend server');
    }
  };

  const redoSearch = async (messageId) => {
    const messageIndex = messages.findIndex((msg) => msg.id === messageId);
    if (messageIndex === -1) return;

    const message = messages[messageIndex];
    if (message.role !== 'user') return;

    setInput(message.content);
    const updatedMessages = messages.slice(0, messageIndex);
    setMessages(updatedMessages);

    const updatedChats = chats.map((chat) =>
      chat.id === selectedChat.id
        ? { ...chat, messages: updatedMessages }
        : chat
    );
    setChats(updatedChats);
  };

  const regenerateResponse = async (messageId) => {
    const messageIndex = messages.findIndex((msg) => msg.id === messageId);
    if (messageIndex === -1) return;

    const message = messages[messageIndex];
    if (message.role !== 'assistant') return;

    await simulateResponseGeneration();

    const newResponse = {
      id: Date.now(),
      role: 'assistant',
      content: `This is a regenerated response from ${selectedModel.name}.`,
      timestamp: new Date().toISOString(),
    };

    const updatedMessages = [...messages.slice(0, messageIndex), newResponse];
    setMessages(updatedMessages);

    const updatedChats = chats.map((chat) =>
      chat.id === selectedChat.id
        ? { ...chat, messages: updatedMessages }
        : chat
    );
    setChats(updatedChats);
  };

  const deleteMessage = (messageId) => {
    const messageIndex = messages.findIndex((msg) => msg.id === messageId);
    if (messageIndex === -1) return;

    const message = messages[messageIndex];
    if (message.role !== 'user') return;

    const updatedMessages = messages.slice(0, messageIndex);
    setMessages(updatedMessages);

    const updatedChats = chats.map((chat) =>
      chat.id === selectedChat.id
        ? { ...chat, messages: updatedMessages }
        : chat
    );
    setChats(updatedChats);
  };

  const createNewChat = () => {
    const newChat = {
      id: Date.now(),
      name: `New Chat ${chats.length + 1}`,
      messages: [],
    };
    setChats([...chats, newChat]);
    setSelectedChat(newChat);
    setMessages([]);
    setActiveTab('chats');
  };

  const deleteChat = (chatId) => {
    const updatedChats = chats.filter((chat) => chat.id !== chatId);
    setChats(updatedChats);

    if (selectedChat.id === chatId) {
      if (updatedChats.length > 0) {
        setSelectedChat(updatedChats[0]);
        setMessages(updatedChats[0].messages);
      } else {
        createNewChat();
      }
    }
  };

  const startEditingChat = (chat) => {
    setEditingChatId(chat.id);
    setEditingChatName(chat.name);
  };

  const saveEditingChat = () => {
    if (editingChatName.trim()) {
      const updatedChats = chats.map((chat) =>
        chat.id === editingChatId
          ? { ...chat, name: editingChatName.trim() }
          : chat
      );
      setChats(updatedChats);
      if (selectedChat.id === editingChatId) {
        setSelectedChat({ ...selectedChat, name: editingChatName.trim() });
      }
    }
    setEditingChatId(null);
    setEditingChatName('');
  };

  const cancelEditingChat = () => {
    setEditingChatId(null);
    setEditingChatName('');
  };

  return (
    <div className="flex h-screen bg-skin-fill">
      {/* Sidebar */}
      <div
        className={`${
          showSidebar ? 'w-64' : 'w-0'
        } bg-skin-fill border-r border-skin-border flex flex-col transition-all duration-300`}
      >
        {/* Tabs */}
        <div className="flex border-b border-skin-border">
          <button
            onClick={() => setActiveTab('chats')}
            className={`flex-1 p-4 text-center ${
              activeTab === 'chats'
                ? 'text-skin-accent border-b-2 border-skin-accent'
                : 'text-skin-muted hover:text-skin-base'
            }`}
          >
            Chats
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`flex-1 p-4 text-center ${
              activeTab === 'settings'
                ? 'text-skin-accent border-b-2 border-skin-accent'
                : 'text-skin-muted hover:text-skin-base'
            }`}
          >
            Settings
          </button>
        </div>

        {/* Content based on active tab */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'chats' ? (
            <div className="h-full flex flex-col">
              <div className="p-4 border-b border-skin-border">
                <button
                  onClick={createNewChat}
                  className="w-full flex items-center justify-center space-x-2 bg-skin-button-accent text-skin-inverted px-4 py-2 rounded-lg hover:bg-skin-button-accent-hover transition-colors"
                >
                  <PlusCircleIcon className="w-5 h-5" />
                  <span>New Chat</span>
                </button>
              </div>
              <div className="flex-1 overflow-y-auto">
                {chats.map((chat) => (
                  <div
                    key={chat.id}
                    className={`group flex items-center justify-between px-4 py-3 hover:bg-skin-fill-hover transition-colors ${
                      selectedChat.id === chat.id ? 'bg-skin-fill-hover' : ''
                    }`}
                  >
                    {editingChatId === chat.id ? (
                      <div className="flex-1 flex items-center space-x-2">
                        <input
                          type="text"
                          value={editingChatName}
                          onChange={(e) => setEditingChatName(e.target.value)}
                          className="flex-1 bg-skin-fill border border-skin-border rounded px-2 py-1 text-skin-base"
                          autoFocus
                        />
                        <button
                          onClick={saveEditingChat}
                          className="text-skin-accent hover:text-skin-base"
                        >
                          <CheckIcon className="w-5 h-5" />
                        </button>
                        <button
                          onClick={cancelEditingChat}
                          className="text-skin-error hover:text-skin-base"
                        >
                          <XMarkIcon className="w-5 h-5" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <button
                          onClick={() => {
                            setSelectedChat(chat);
                            setMessages(chat.messages);
                          }}
                          className="flex-1 flex items-center space-x-2 text-left text-skin-base"
                        >
                          <ChatBubbleLeftIcon className="w-5 h-5" />
                          <span className="truncate">{chat.name}</span>
                        </button>
                        <div className="flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              startEditingChat(chat);
                            }}
                            className="text-skin-muted hover:text-skin-base"
                          >
                            <PencilIcon className="w-5 h-5" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteChat(chat.id);
                            }}
                            className="text-skin-muted hover:text-skin-error"
                          >
                            <TrashIcon className="w-5 h-5" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-full overflow-y-auto">
              <div className="p-4 space-y-6 text-skin-base">
                <div className="relative h-32 mb-6 flex items-center justify-center">
                  <img
                    src={settingsImage}
                    alt="Settings theme"
                    className="h-[128px] w-auto object-contain"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-4">Model Settings</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Model
                      </label>
                      <select
                        value={selectedModel.id}
                        onChange={(e) =>
                          setSelectedModel(
                            models.find((m) => m.id === e.target.value)
                          )
                        }
                        className="w-full bg-skin-fill-hover border border-skin-border rounded-md px-3 py-2 text-skin-base"
                      >
                        {models.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        API Key
                      </label>
                      <input
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="Enter API key"
                        className="w-full bg-skin-fill-hover border border-skin-border rounded-md px-3 py-2 text-skin-base placeholder-skin-muted"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Search Length
                      </label>
                      <input
                        type="number"
                        value={searchLength}
                        onChange={(e) =>
                          setSearchLength(
                            Math.max(100, parseInt(e.target.value))
                          )
                        }
                        min="100"
                        className="w-full bg-skin-fill-hover border border-skin-border rounded-md px-3 py-2 text-skin-base"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Extension Size
                      </label>
                      <input
                        type="number"
                        value={extensionSize}
                        onChange={(e) =>
                          setExtensionSize(
                            Math.max(50, parseInt(e.target.value))
                          )
                        }
                        min="50"
                        className="w-full bg-skin-fill-hover border border-skin-border rounded-md px-3 py-2 text-skin-base"
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4">Appearance</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Theme
                      </label>
                      <select
                        value={theme.id}
                        onChange={(e) =>
                          setTheme(themes.find((t) => t.id === e.target.value))
                        }
                        className="w-full bg-skin-fill-hover border border-skin-border rounded-md px-3 py-2 text-skin-base"
                      >
                        {themes.map((t) => (
                          <option key={t.id} value={t.id}>
                            {t.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Font Size
                      </label>
                      <select
                        value={fontSize.id}
                        onChange={(e) =>
                          setFontSize(
                            fontSizes.find((f) => f.id === e.target.value)
                          )
                        }
                        className="w-full bg-skin-fill-hover border border-skin-border rounded-md px-3 py-2 text-skin-base"
                      >
                        {fontSizes.map((f) => (
                          <option key={f.id} value={f.id}>
                            {f.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Project Title */}
        <div className="bg-skin-fill border-b border-skin-border p-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-skin-base">
            CrossBar-GraphRAG
          </h1>
          <div className="flex items-center space-x-2">
            <PDFDownloadLink
              document={<ChatPDF messages={messages} selectedModel={selectedModel} />}
              fileName={`${selectedChat.name}.pdf`}
              className="p-2 rounded-lg bg-skin-fill-hover shadow-md hover:bg-skin-button-muted text-skin-base"
            >
              {({ loading }) => (
                loading ? (
                  <ArrowPathIcon className="w-6 h-6 animate-spin" />
                ) : (
                  <DocumentArrowDownIcon className="w-6 h-6" />
                )
              )}
            </PDFDownloadLink>
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="p-2 rounded-lg bg-skin-fill-hover shadow-md hover:bg-skin-button-muted text-skin-base"
            >
              <Cog6ToothIcon className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-skin-fill">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[70%] rounded-lg p-4 ${
                  message.role === 'user'
                    ? 'bg-skin-button-accent text-skin-inverted'
                    : 'bg-skin-fill-hover border border-skin-border text-skin-base'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <ChatBubbleLeftIcon className="w-5 h-5" />
                    <span className="font-medium">
                      {message.role === 'user' ? 'You' : selectedModel.name}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    {message.role === 'user' ? (
                      <>
                        <button
                          onClick={() => redoSearch(message.id)}
                          className="text-skin-inverted hover:text-skin-error transition-colors"
                          title="Redo search with current parameters"
                        >
                          <ArrowPathIcon className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => deleteMessage(message.id)}
                          className="text-skin-inverted hover:text-skin-error transition-colors"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={() => regenerateResponse(message.id)}
                        className="text-skin-base hover:text-skin-accent transition-colors"
                        title="Regenerate response"
                      >
                        <ArrowPathIcon className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
                <ReactMarkdown
                  className={`prose prose-${theme.id} prose-${fontSize.id} max-w-none`}
                >
                  {message.content}
                </ReactMarkdown>
                <div className="text-xs opacity-70 mt-2">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          {isGenerating && (
            <div className="flex justify-start">
              <div className="max-w-[70%] rounded-lg p-4 bg-skin-fill-hover border border-skin-border text-skin-base">
                <div className="flex items-center space-x-2">
                  <ArrowPathIcon className="w-5 h-5 animate-spin" />
                  <span>{generationStep}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-skin-border p-4 bg-skin-fill">
          <form onSubmit={handleSubmit} className="flex flex-col space-y-2">
            {tokenError && (
              <div className="text-skin-error text-sm">{tokenError}</div>
            )}
            <div className="flex space-x-4">
              <input
                type="text"
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  setTokenError('');
                }}
                placeholder="Type your message..."
                className="flex-1 bg-skin-fill-hover border border-skin-border rounded-lg px-4 py-2 text-skin-base placeholder-skin-muted focus:outline-none focus:ring-2 focus:ring-skin-accent focus:border-transparent"
              />
              <button
                type="submit"
                disabled={isGenerating}
                className={`bg-skin-button-accent text-skin-inverted px-4 py-2 rounded-lg hover:bg-skin-button-accent-hover transition-colors flex items-center space-x-2 ${
                  isGenerating ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                <span>Send</span>
                <PaperAirplaneIcon className="w-5 h-5" />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
