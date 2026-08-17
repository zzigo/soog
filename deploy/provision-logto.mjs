import { readFile, rename, writeFile } from 'node:fs/promises'
import { spawn } from 'node:child_process'

const endpoint = process.env.SOOG_LOGTO_ENDPOINT || 'https://logto.zztt.org'
const adminEndpoint = process.env.SOOG_LOGTO_ADMIN_ENDPOINT || 'https://logto-admin.zztt.org'
const backendEnvPath = process.env.SOOG_BACKEND_ENV_PATH || '/opt/soog/backend/.env'
const frontendEnvPath = process.env.SOOG_FRONTEND_ENV_PATH || '/opt/soog/frontend/.env'
const managementResource = 'https://default.logto.app/api'
const apiResource = 'https://soog.zztt.org/api'
const productionOrigin = 'https://soog.zztt.org'
const localOrigin = 'http://localhost:3000'
const adminSubjects = process.env.SOOG_ADMIN_SUBJECTS || 'b2rx7bymuotq,30iyovrlwm2q'

async function dockerNode(source) {
  return new Promise((resolve, reject) => {
    const child = spawn('docker', ['exec', '-i', 'logto', 'node'], {
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout.setEncoding('utf8').on('data', (value) => { stdout += value })
    child.stderr.setEncoding('utf8').on('data', (value) => { stderr += value })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Logto container query failed: ${stderr.trim() || `exit ${code}`}`))
        return
      }
      try {
        resolve(JSON.parse(stdout))
      } catch {
        reject(new Error('Logto container returned an invalid response'))
      }
    })
    child.stdin.end(source)
  })
}

async function managementCredentials() {
  return dockerNode(String.raw`
    (async()=>{
      const {Client}=require('pg');
      const {randomBytes}=require('crypto');
      const client=new Client({connectionString:process.env.DB_URL});
      await client.connect();
      let result=await client.query(
        "select a.id,s.value as secret from applications a join application_secrets s on s.application_id=a.id where a.id='m-default' and (s.expires_at is null or s.expires_at>now()) order by s.created_at desc limit 1"
      );
      if(!result.rows[0]){
        const secret=randomBytes(24).toString('base64url');
        await client.query(
          "insert into application_secrets(tenant_id,application_id,name,value,created_at,expires_at) select tenant_id,id,'SOOG provisioning',$1,now(),null from applications where id='m-default' on conflict do nothing",
          [secret]
        );
        result=await client.query(
          "select a.id,s.value as secret from applications a join application_secrets s on s.application_id=a.id where a.id='m-default' and (s.expires_at is null or s.expires_at>now()) order by s.created_at desc limit 1"
        );
      }
      await client.end();
      if(!result.rows[0]) throw new Error('Management application secret not found');
      console.log(JSON.stringify({id:result.rows[0].id,secret:result.rows[0].secret}));
    })().catch(error=>{console.error(error.message);process.exit(1)});
  `)
}

async function api(path, token, init = {}) {
  const response = await fetch(`${endpoint}${path}`, {
    ...init,
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/json',
      ...init.headers,
    },
  })
  if (!response.ok) {
    const detail = (await response.text()).replace(/\s+/g, ' ').slice(0, 300)
    throw new Error(`Logto Management API ${path} failed with HTTP ${response.status}: ${detail}`)
  }
  return response.json()
}

function setEnv(source, key, value) {
  const line = `${key}=${value}`
  const expression = new RegExp(`^${key}=.*$`, 'm')
  return expression.test(source)
    ? source.replace(expression, line)
    : `${source.trimEnd()}\n${line}\n`
}

async function updateEnv(path, values) {
  let source = ''
  try {
    source = await readFile(path, 'utf8')
  } catch (error) {
    if (error.code !== 'ENOENT') throw error
  }
  for (const [key, value] of Object.entries(values)) source = setEnv(source, key, value)
  const temporaryPath = `${path}.logto-${process.pid}`
  await writeFile(temporaryPath, source, { mode: 0o600 })
  await rename(temporaryPath, path)
}

const management = await managementCredentials()
const tokenResponse = await fetch(`${adminEndpoint}/oidc/token`, {
  method: 'POST',
  headers: {
    authorization: `Basic ${Buffer.from(`${management.id}:${management.secret}`).toString('base64')}`,
    'content-type': 'application/x-www-form-urlencoded',
  },
  body: new URLSearchParams({
    grant_type: 'client_credentials',
    resource: managementResource,
    scope: 'all',
  }),
})
if (!tokenResponse.ok) {
  const detail = (await tokenResponse.text()).replace(/\s+/g, ' ').slice(0, 300)
  throw new Error(`Logto Management token failed with HTTP ${tokenResponse.status}: ${detail}`)
}
const { access_token: accessToken } = await tokenResponse.json()
if (!accessToken) throw new Error('Logto Management token response omitted access_token')

const applications = await api('/api/applications?page=1&page_size=100', accessToken)
let application = applications.find((candidate) => candidate.name.toLowerCase() === 'soog')
const applicationConfig = {
  name: 'soog',
  description: 'SOOG organogram generator',
  oidcClientMetadata: {
    redirectUris: [`${productionOrigin}/auth/callback`, `${localOrigin}/auth/callback`],
    postLogoutRedirectUris: [`${productionOrigin}/`, `${localOrigin}/`],
    backchannelLogoutSessionRequired: false,
  },
  customClientMetadata: {
    idTokenTtl: 3600,
    allowTokenExchange: false,
    corsAllowedOrigins: [productionOrigin, localOrigin],
    rotateRefreshToken: true,
    refreshTokenTtlInDays: 14,
    alwaysIssueRefreshToken: false,
  },
  customData: { boundedContext: 'soog' },
}
if (!application) {
  application = await api('/api/applications', accessToken, {
    method: 'POST',
    body: JSON.stringify({ ...applicationConfig, type: 'SPA' }),
  })
} else {
  application = await api(`/api/applications/${application.id}`, accessToken, {
    method: 'PATCH',
    body: JSON.stringify(applicationConfig),
  })
}
if (application.type !== 'SPA') throw new Error('Existing SOOG Logto application is not a SPA')

const resources = await api('/api/resources?page=1&page_size=100', accessToken)
let resource = resources.find((candidate) => candidate.indicator === apiResource)
const resourceConfig = {
  name: 'SOOG API',
  indicator: apiResource,
  accessTokenTtl: 3600,
}
if (!resource) {
  resource = await api('/api/resources', accessToken, {
    method: 'POST',
    body: JSON.stringify(resourceConfig),
  })
} else {
  resource = await api(`/api/resources/${resource.id}`, accessToken, {
    method: 'PATCH',
    body: JSON.stringify(resourceConfig),
  })
}

await updateEnv(backendEnvPath, {
  LOGTO_ISSUER_URL: `${endpoint}/oidc`,
  LOGTO_SIGNING_ALGORITHMS: 'ES384',
  SOOG_LOGTO_APP_ID: application.id,
  SOOG_LOGTO_API_RESOURCE: apiResource,
  SOOG_ADMIN_SUBJECTS: adminSubjects,
  SOOG_DEFAULT_DAILY_RENDER_QUOTA: '10',
  SOOG_DEFAULT_WEEKLY_RENDER_QUOTA: '40',
  SOOG_USAGE_TIMEZONE: 'Europe/Zurich',
  FLASK_DEBUG: '0',
  FLASK_RELOAD: '0',
})
await updateEnv(frontendEnvPath, {
  NUXT_PUBLIC_LOGTO_ENDPOINT: endpoint,
  NUXT_PUBLIC_LOGTO_APP_ID: application.id,
  NUXT_PUBLIC_LOGTO_API_RESOURCE: apiResource,
})

console.info(JSON.stringify({
  status: 'configured',
  applicationId: application.id,
  applicationType: application.type,
  resourceId: resource.id,
  resourceIndicator: resource.indicator,
  adminSubjects: adminSubjects.split(',').filter(Boolean).length,
}))
