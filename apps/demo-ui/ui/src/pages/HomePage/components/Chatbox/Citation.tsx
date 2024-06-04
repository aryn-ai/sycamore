import { useState } from "react";
import { SearchResultDocument } from "../../../../Types";
import {
  IconFileTypeHtml,
  IconFileTypePdf,
  IconLink,
} from "@tabler/icons-react";
import {
  Anchor,
  Badge,
  Group,
  HoverCard,
  Text,
  useMantineTheme,
} from "@mantine/core";

export const Citation = ({
  document,
  citationNumber,
}: {
  document: SearchResultDocument;
  citationNumber: number;
}) => {
  const [doc, setDoc] = useState(document);
  const [docId, setDocId] = useState(document.id);
  const [docUrl, setDocUrl] = useState(document.url);
  const [citNum, setCitNum] = useState(citationNumber);
  const theme = useMantineTheme();
  function icon() {
    if (document.isPdf()) {
      return <IconFileTypePdf size="1.125rem" color={theme.colors.blue[6]} />;
    } else if (document.url.endsWith("htm") || document.url.endsWith("html")) {
      return <IconFileTypeHtml size="1.125rem" color={theme.colors.blue[6]} />;
    }
    return <IconLink size="1.125rem" />;
  }
  return (
    <HoverCard shadow="sm">
      <HoverCard.Target>
        <Anchor
          key={docId + Math.random()}
          fz="xs"
          target="_blank"
          style={{ verticalAlign: "super" }}
          onClick={(event) => {
            event.preventDefault();
            if (doc.isPdf()) {
              const dataString = JSON.stringify(doc);
              localStorage.setItem("pdfDocumentMetadata", dataString);
              window.open("/viewPdf");
            } else {
              window.open(docUrl);
            }
          }}
        >
          <Badge size="xs" color="gray" variant="filled">
            {citNum}
          </Badge>
        </Anchor>
      </HoverCard.Target>
      <HoverCard.Dropdown>
        <Group>
          <Text size="xs">{doc.title}</Text>
          {icon()}
        </Group>
        <Text size="xs" c={theme.colors.gray[6]}>
          {" "}
          {doc.url}
        </Text>
      </HoverCard.Dropdown>
    </HoverCard>
  );
};
