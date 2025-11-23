---
layout: page
title: Talks
permalink: /talks/
---

<ul class="post-list">
  {% for talk in site.talks reversed %}
    <li>
      {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
      <span class="post-meta">{{ talk.date | date: date_format }}</span>
      <h3>
        <a class="post-link" href="{{ talk.url | relative_url }}">
          {{ talk.title | escape }}
        </a>
      </h3>
    </li>
  {% endfor %}
</ul>