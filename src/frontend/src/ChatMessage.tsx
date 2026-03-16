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
                {isUser ? '👤' : '🤖'}
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
