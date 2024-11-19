import React, { useState } from 'react'

import {
  actions,
  AppStatus,
  thunkActions,
  useAppDispatch,
  useAppSelector,
} from 'store/provider'
import { Header } from 'components/header'
import { Chat } from 'components/chat/chat'
import SearchInput from 'components/search_input'
import { ReactComponent as ChatIcon } from 'images/chat_icon.svg'
import { ReactComponent as ElasticLogo } from 'images/elastic_logo.svg'
import IndexLogo from 'images/index_logo.png'
import { SearchResults } from './components/search_results'

const App = () => {
  const dispatch = useAppDispatch()
  const status = useAppSelector((state) => state.status)
  const sources = useAppSelector((state) => state.sources)
  const [summary, ...messages] = useAppSelector((state) => state.conversation)
  const hasSummary = useAppSelector(
    (state) => !!state.conversation?.[0]?.content
  )
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [index, setIndex] = useState<string>('')

  const handleSearch = (query: string) => {
    dispatch(thunkActions.search(query, index))
  }

  const handleSendChatMessage = (query: string) => {
    dispatch(thunkActions.askQuestion(query, index))
  }
  const handleAbortRequest = () => {
    dispatch(thunkActions.abortRequest())
  }
  const handleToggleSource = (name) => {
    dispatch(actions.sourceToggle({ name }))
  }
  const handleSourceClick = (name) => {
    dispatch(actions.sourceToggle({ name, expanded: true }))

    setTimeout(() => {
      document
        .querySelector(`[data-source="${name}"]`)
        ?.scrollIntoView({ behavior: 'smooth' })
    }, 300)
  }

  const suggestedQueries = [
    "What is this new AutoOps feature I keep hearing about?",
    "Is Elastic or OpenSearch more performant?",
    "How can I ingest data from an s3 bucket into Elastic?"
  ]

  const indexes = [
    'elastic-search-labs',
    'search-elastic-blogs',
  ]

  const handleDropdownChange = (e) => {
    const index = e.target.value
    setIndex(index)
  }

  return (
    <>
      <Header />

      <div className="p-4 max-w-2xl mx-auto">
        {/* Dropdown -- SHOULD BE MOVED TO SEPARATE CLASS */}
        <div className="relative inline-block w-full mt-4 mb-12">
          <h2 className="text-zinc-400 text-sm font-medium mb-3  inline-flex items-center gap-2">
            <img src={IndexLogo} alt="Index Logo" className="w-10 h-5" /> Index
          </h2>
          <select
            className="block w-full h-12 px-4 py-2 bg-blue-200 rounded-md shadow text-zinc-700"
            value={index}
            onChange={handleDropdownChange}
          >
            <option value="" disabled>Select an index</option>
            {indexes.map((index) => (
              <option key={index} value={index}>
                {index}
              </option>
            ))}
          </select>
        </div>

        <SearchInput
          onSearch={handleSearch}
          value={searchQuery}
          appStatus={status}
        />

        {status === AppStatus.Idle ? (
          <div className="mx-auto my-6">
            <h2 className="text-zinc-400 text-sm font-medium mb-3  inline-flex items-center gap-2">
              <ChatIcon /> Common questions
            </h2>
            <div className="flex flex-col space-y-4">
              {suggestedQueries.map((query) => (
                <button
                  key={query}
                  className="hover:-translate-y-1 hover:shadow-lg hover:bg-zinc-300 transition-transform h-12 px-4 py-2 bg-zinc-200 rounded-md shadow flex items-center text-zinc-700"
                  onClick={(e) => {
                    e.preventDefault()
                    setSearchQuery(query)
                    handleSearch(query)
                  }}
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {hasSummary ? (
              <div className="max-w-2xl mx-auto relative">
                <Chat
                  status={status}
                  messages={messages}
                  summary={summary}
                  onSend={handleSendChatMessage}
                  onAbortRequest={handleAbortRequest}
                  onSourceClick={handleSourceClick}
                />

                <SearchResults
                  results={sources}
                  toggleSource={handleToggleSource}
                />
              </div>
            ) : (
              <div className="h-36 p-6 bg-white rounded-md shadow flex flex-col justify-start items-center gap-4 mt-6">
                <ElasticLogo className="w-16 h-16" />
                <p className="text-center text-zinc-400 text-sm ">
                  Looking that up for you...
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </>
  )
}

export default App
