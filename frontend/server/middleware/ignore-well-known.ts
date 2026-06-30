import { defineEventHandler, setResponseStatus } from 'h3'

export default defineEventHandler((event) => {
  const url = event.node.req.url || ''
  if (url.includes('/.well-known/appspecific/')) {
    setResponseStatus(event, 404)
    return ''
  }
})
