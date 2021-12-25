import * as React from "react"

import "../styles/prompt.scss"

const Prompt = ({ children, command }) => {


  return (
    <p className="prompt"><span className="prompt-symbol">{">"} </span><span className="prompt-command">{command} </span><h2>{children}</h2></p>
  )
}

export default Prompt
