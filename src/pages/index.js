import React from "react"

import LandingBio from "../components/landing-bio"
import Layout from "../components/layout"
import SEO from "../components/seo"

const IndexPage = () => (
  <Layout>
    <SEO 
    title="Home" 
    keywords={['sergi', 'masip', 'sergi masip', 'contant', 'software', 'data', 'web', 'engineer', 'developer', 'machine learning', 'artifical intelligence', 'unity']} 
    />
    <LandingBio />
  </Layout>
)

export default IndexPage
