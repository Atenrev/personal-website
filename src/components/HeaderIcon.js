import * as React from "react"
import { StaticImage } from "gatsby-plugin-image"

// import "../styles/prompt.scss"

const HeaderIcon = ({ icon }) => {
  const imageSrc = `../assets/${icon}.png`;

  return (
    <div>
      <StaticImage
        src={imageSrc}
        alt="Icon."
        // height="24px"
      />
    </div>
  )
}

export default HeaderIcon
