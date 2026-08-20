"""BasicAuthProvider —— 用户名/密码仪表盘身份验证（无需 OAuth IDP）。

一种适用于自托管场景的“直接为仪表盘设置密码”的提供者。
它接入了与 Nous OAuth 提供者相同的 ``DashboardAuthProvider`` 框架，
但通过“用户名 + 密码”而非 OAuth 重定向来进行身份验证：
它设置了 ``supports_password = True`` 并实现了 ``complete_password_login``。
登录页面会为其渲染一个凭据表单；
登录之后的所有下游流程（Session Cookie、校验、刷新、WebSocket Token、登出）
均与 OAuth 路径完全一致，因为密码 Session 本质上就是一个
包含提供者自行签发的不透明 Token 的 :class:`Session` 对象。

该提供者**无需外部 IDP，也无需数据库**。
凭据需要预先配置；
Session 则是由该提供者自身签发并校验的无状态 HMAC 签名 Token。
这使其保持了零基础设施依赖 —— 非常适合单机自托管仪表盘。

配置项（环境变量在设为非空值时优先级高于 config.yaml），
遵循与 Nous 提供者相同的优先级规范：

  ``config.yaml`` —— 规范配置入口：:

      dashboard:
        basic_auth:
          username: admin               # 必填
          # 提供 预先计算好的 scrypt 哈希值（推荐 —— 静态存储时不含明文）...
          password_hash: "scrypt$..."   # 参见 hash_password()
          # ... 或 提供明文密码（加载时会在内存中进行哈希）。
          password: "s3cret"
          secret: "<32+ 字节随机序列，base64 或 hex 格式>"  # 选填；用于签名 Token 的密钥
          session_ttl_seconds: 43200    # 选填；访问令牌生命周期（默认 12 小时）

  环境变量重写：:

      HERMES_DASHBOARD_BASIC_AUTH_USERNAME
      HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH   # 推荐
      HERMES_DASHBOARD_BASIC_AUTH_PASSWORD        # 明文备用路径
      HERMES_DASHBOARD_BASIC_AUTH_SECRET
      HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS

若未配置 ``secret``，系统将在启动时为每个进程生成一个随机 Secret。
这对于单进程仪表盘来说没有问题，但意味着所有 Session 会在重启时失效，
且无法在多个 Worker 进程间共享 ——
如需实现稳定、跨 Worker 或重启后依然有效的 Session，请显式配置 ``secret``。

密码哈希采用标准库中的 :func:`hashlib.scrypt`（内存硬函数，无第三方依赖）。
``complete_password_login`` 采用常数时间比较，
且即使用户名不存在也始终会执行一次哈希，
因此该端点不会构成通过时序攻击进行用户名枚举的漏洞。

跳过原因（Skip reasons）：
  与 Nous 提供者类似，本模块暴露了一个模块级的 ``LAST_SKIP_REASON``，
  以便当插件已加载但拒绝注册时（未配置用户名/密码），
  网关的“安全关闭（fail-closed）”分支能够向外呈现提示信息。
"""