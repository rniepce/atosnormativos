import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessageProps } from './types';

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
    const isUser = message.role === 'user';
    const roleClass = isUser ? 'user' : 'assistant';
    const [sourcesExpanded, setSourcesExpanded] = useState(false);

    return (
        <div className={`chat-message ${roleClass}`}>
            <div className={`chat-avatar ${roleClass}`}>
                {isUser ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                )}
            </div>

            <div className="chat-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                </ReactMarkdown>

                {message.sources && message.sources.length > 0 && (
                    <div className="sources-expander">
                        <div
                            className="sources-header"
                            onClick={() => setSourcesExpanded(!sourcesExpanded)}
                        >
                            <span>📚 Fontes Consultadas ({message.sources.length})</span>
                            <span>{sourcesExpanded ? '▼' : '▶'}</span>
                        </div>

                        {sourcesExpanded && (
                            <div className="sources-content">
                                {message.sources.map((src, idx) => {
                                    let badgeClass = 'badge-unknown';
                                    if (src.status === 'VIGENTE') badgeClass = 'badge-vigente';
                                    if (src.status === 'REVOGADO') badgeClass = 'badge-revogado';

                                    return (
                                        <div className="source-card" key={idx}>
                                            <div className="source-title">
                                                {src.tipo} {src.numero}/{src.ano}
                                                {src.status && (
                                                    <span className={`badge ${badgeClass}`}>
                                                        {src.status}
                                                    </span>
                                                )}
                                                {src.score !== undefined && (
                                                    <span className="source-score">
                                                        Score: {src.score.toFixed(2)}
                                                    </span>
                                                )}
                                            </div>
                                            <div className="source-excerpt">
                                                {src.chunk_text ? src.chunk_text.substring(0, 300) + '…' : ''}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ChatMessage;
