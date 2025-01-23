<template>
  <div>
    <!-- <nav class="nav">
      <NuxtLink to="/" class="nav-link">Home</NuxtLink>
      <NuxtLink to="/logs" class="nav-link">Logs</NuxtLink>
    </nav> -->
    <NuxtPage />
  </div>
</template>

<style>
.nav {
  background: #1a1a1a;
  padding: 1rem;
  margin-bottom: 2rem;
}

.nav-link {
  color: #fff;
  text-decoration: none;
  margin-right: 1rem;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.nav-link:hover {
  background: rgba(255, 255, 255, 0.1);
}

.nav-link.router-link-active {
  background: #4caf50;
}
</style>
